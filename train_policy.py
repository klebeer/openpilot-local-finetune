"""Fine-tune a split driving model's policy on the stretches you drove, vision frozen.

Inputs come from replay_split.py's chain-mode outputs: each sample is what the policy sees on the comma, the vision
hidden states of the last 25 chain frames (4 frames apart, the current one last), zero desire, a left-hand-drive car.

Labels follow openpilot.distill's plan targets, built from the localizer but in the device's own calibrated frame (the
frame the model saw): lateral position, heading change (yaw) and yaw rate at the plan's times up to LABEL_S. Steering
reads yaw and yaw rate, so all three are fitted. They count on frames where you steered through the next WHO_WINDOW_S
with the car moving at MAX_LABEL_KPH or slower; faster ones of yours are anchored like openpilot's. The loss is the L1 to your path, each quantity scaled by its spread, plus the L1 between every
raw output and the original policy's: ANCHOR on all frames, ANCHOR + ANCHOR_OP where openpilot steered through the
window or you steered above MAX_LABEL_KPH, so what the labels do not cover stays as it was and openpilot's own
driving least of all.

Segments are read from $LOCROOT/data (default ~/.comma/locroot). Segments on the held-out list are never used. VAL_SHARE of the remaining drives (whole drives) picks the epoch: the
lowest error to your path, plus OP_PENALTY times any error to the real path where openpilot steered beyond the
original policy's. Writes the best state_dict and, for the held-out segments of the first replay dir, replay files
with the fine-tuned plan (the original policy's run through the same torch module is written beside them, to
separate conversion from training).

Usage: python train_policy.py <model dir> <replay dir>[,<replay dir> ...] <held-out list> <out dir>
"""
import os
import sys
from bisect import bisect_right

import numpy as np
import torch
from safetensors.numpy import load
from scipy.ndimage import gaussian_filter1d

from openpilot.common.transformations.orientation import euler_from_rot, rot_from_euler, rot_from_quat
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.tools.lib.logreader import LogReader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stock_to_torch  # noqa: E402,F401  (registers the converters onnx2torch lacks)
from onnx2torch import convert  # noqa: E402
import onnx  # noqa: E402

T_IDXS = np.asarray(ModelConstants.T_IDXS)
LABEL_S = 3.2
N_LABEL = int(np.searchsorted(T_IDXS, LABEL_S))   # plan times 0 .. 3.16 s
WHO_WINDOW_S = 3.0
MIN_SPEED = 3.0
MAX_LABEL_KPH = 70.0   # above this you mostly steer to overtake or avoid, not to hold your place in the lane
HISTORY = 25
FEATURES = 512
PLAN_N, PLAN_W = 33, 15
Y, YAW, YAW_RATE = 1, 11, 14                       # Plan.POSITION y, T_FROM_CURRENT_EULER z, ORIENTATION_RATE z
ANCHOR = 1.0
ANCHOR_OP = 3.0
OP_PENALTY = 10.0
VAL_SHARE = 0.1
EPOCHS, BATCH, LR = 10, 512, 1e-4
NONE, YOURS, OPENPILOT, FAST = 0, 1, 2, 3   # FAST: yours, above MAX_LABEL_KPH


def log_timelines(rlog):
  calib, lat = [], []
  for m in LogReader(rlog):
    w = m.which()
    if w == "extrinsicsCalibration":
      calib.append((m.logMonoTime * 1e-9, np.array(m.extrinsicsCalibration.rpyCalib)))
    elif w == "carControl":
      lat.append((m.logMonoTime * 1e-9, m.carControl.latActive))
  return calib, lat


def at(timeline, t):
  i = bisect_right([x[0] for x in timeline], t) - 1
  return timeline[max(i, 0)][1]


def targets(states, frame_t, fidx, rpy):
  """Lateral position, yaw and yaw rate at T_IDXS[:N_LABEL] from frame fidx, in the calibrated frame."""
  query = T_IDXS[:N_LABEL] + frame_t[fidx]
  i = np.clip(np.searchsorted(frame_t, query) - 1, 0, len(frame_t) - 2)
  w = ((query - frame_t[i]) / (frame_t[i + 1] - frame_t[i]))[:, None]
  fut = states[i] * (1 - w) + states[i + 1] * w
  quats = fut[:, 3:7] / np.linalg.norm(fut[:, 3:7], axis=1, keepdims=True)
  calib_from_device = rot_from_euler(rpy).T
  calib_from_ecef = np.einsum("ij,tjk->tik", calib_from_device, rot_from_quat(quats).swapaxes(1, 2))
  pos = np.einsum("ij,tj->ti", calib_from_ecef[0], fut[:, 0:3] - fut[0, 0:3])
  euler = euler_from_rot(np.einsum("ij,tjk->tik", calib_from_ecef[0], calib_from_ecef.swapaxes(1, 2)))
  rate = np.einsum("ij,tj->ti", calib_from_device, fut[:, 10:13])
  return np.stack([pos[:, 1], euler[:, 2], rate[:, 2]], axis=1).astype(np.float32)   # (N_LABEL, 3)


def segment_samples(seg_dir, replay_path):
  """Hidden states with HISTORY-1 zero rows in front, and per chain frame the real path and who steered (or None)."""
  r = np.load(replay_path)
  hidden = np.concatenate([np.zeros((HISTORY - 1, FEATURES), np.float16), r["hidden"]])
  labels = [None] * len(r["t_eof"])
  if os.path.exists(f"{seg_dir}/localizer.safetensors"):
    loc = load(open(f"{seg_dir}/localizer.safetensors", "rb").read())
    info = load(open(f"{seg_dir}/frame_info.safetensors", "rb").read())
    frame_t, states = info["fcamera/t"], loc["frame_states"].copy()
    states[:, 10:13] = gaussian_filter1d(states[:, 10:13], 2, axis=0, radius=5, mode="nearest")
    calib, lat = log_timelines(f"{seg_dir}/rlog.zst")
    for k, t_eof in enumerate(r["t_eof"]):
      t = t_eof * 1e-9 - 0.008
      fidx = int(np.argmin(np.abs(frame_t - t)))
      if abs(frame_t[fidx] - t) > 0.02 or frame_t[-1] < frame_t[fidx] + LABEL_S or not calib:
        continue
      if float(np.linalg.norm(states[fidx, 7:10])) < MIN_SPEED:
        continue
      win = [x for x in lat if t <= x[0] <= t + WHO_WINDOW_S]
      who = NONE if not win else YOURS if not any(x[1] for x in win) else OPENPILOT if all(x[1] for x in win) else NONE
      if who == YOURS and float(np.linalg.norm(states[fidx, 7:10])) * 3.6 > MAX_LABEL_KPH:
        who = FAST
      if who != NONE:
        labels[k] = (who, targets(states, frame_t, fidx, at(calib, t)))
  return hidden, labels


def load_policy(model_dir):
  tm = convert(onnx.load(os.path.join(model_dir, "driving_policy.onnx")))
  for name, buf in list(tm.initializers.named_buffers()):
    if buf.is_floating_point():
      del tm.initializers._buffers[name]
      tm.initializers.register_parameter(name, torch.nn.Parameter(buf.float()))
  return tm.float()


class Data:
  def __init__(self, segs, model_dir_unused=None):
    self.hidden, self.index, self.labels, self.who = [], [], [], []
    base = 0
    for seg_dir, replay_path in segs:
      hidden, labels = segment_samples(seg_dir, replay_path)
      self.hidden.append(hidden)
      for k, lab in enumerate(labels):
        self.index.append(base + k)
        self.who.append(lab[0] if lab is not None else NONE)
        self.labels.append(lab[1] if lab is not None else np.zeros((N_LABEL, 3), np.float32))
      base += len(hidden)
    self.hidden = torch.from_numpy(np.concatenate(self.hidden).astype(np.float32)) if self.hidden else torch.zeros((0, FEATURES))
    self.index = torch.tensor(self.index, dtype=torch.long)
    self.labels = torch.from_numpy(np.array(self.labels)) if self.labels else torch.zeros((0, N_LABEL, 3))
    self.who = torch.tensor(self.who, dtype=torch.long)
    self.labelled = self.who == YOURS

  def __len__(self):
    return len(self.index)

  def batch(self, rows, device):
    starts = self.index[rows]
    feats = self.hidden[starts[:, None] + torch.arange(HISTORY)[None]]
    return feats.to(device), self.labels[rows].to(device), self.who[rows].to(device)


def run(model, feats):
  """The converted graph reshapes to a batch of one, so each sample is mapped through it."""
  def one(d, t, f):
    out = model(d[None], t[None], f[None])
    return (out[0] if isinstance(out, (list, tuple)) else out)[0]
  n = feats.shape[0]
  desire = torch.zeros((n, HISTORY, 8), device=feats.device)
  traffic = torch.tensor([[1.0, 0.0]], device=feats.device).expand(n, 2)
  return torch.func.vmap(one)(desire, traffic, feats)


def lateral(out):
  plan = out[:, :PLAN_N * PLAN_W].reshape(-1, PLAN_N, PLAN_W)[:, :N_LABEL]
  return torch.stack([plan[..., Y], plan[..., YAW], plan[..., YAW_RATE]], dim=-1)


def evaluate(model, data, device):
  """Mean L1 to the real path (y, yaw, yaw rate) where you steered and where openpilot steered."""
  model.eval()
  err, n = {YOURS: torch.zeros(3), OPENPILOT: torch.zeros(3)}, {YOURS: 0, OPENPILOT: 0}
  with torch.no_grad():
    for start in range(0, len(data), 2048):
      rows = torch.arange(start, min(start + 2048, len(data)))
      feats, labels, who = data.batch(rows, device)
      pred = lateral(run(model, feats))
      for kind in (YOURS, OPENPILOT):
        mask = who == kind
        if mask.any():
          err[kind] += (pred[mask] - labels[mask]).abs().mean(dim=1).sum(0).cpu(); n[kind] += int(mask.sum())
  return {kind: (err[kind] / max(n[kind], 1)).numpy() for kind in err}


def main(model_dir, replay_dirs, held_out_list, out_dir):
  os.makedirs(out_dir, exist_ok=True)
  device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
  held = {os.path.basename(line.strip().rstrip("/")) for line in open(held_out_list) if line.strip()}
  loc_root = os.path.join(os.environ.get("LOCROOT", os.path.expanduser("~/.comma/locroot")), "data")
  replay_dirs = replay_dirs.split(",")
  segs = sorted(f[:-4] for f in os.listdir(replay_dirs[0]) if f.endswith(".npz"))
  pairs = {s: (os.path.join(loc_root, s), os.path.join(replay_dirs[0], s + ".npz")) for s in segs}
  runs = [(s, (os.path.join(loc_root, s), os.path.join(d, s + ".npz"))) for d in replay_dirs for s in segs
          if os.path.exists(os.path.join(d, s + ".npz"))]
  train_names = [s for s in segs if s not in held]
  drives = sorted({s.split("--")[0] for s in train_names})
  rng = np.random.default_rng(0)
  val_drives = set(rng.choice(drives, max(1, int(len(drives) * VAL_SHARE)), replace=False))
  train = Data([p for s, p in runs if s not in held and s.split("--")[0] not in val_drives])
  val = Data([p for s, p in runs if s not in held and s.split("--")[0] in val_drives])
  print(f"train {len(train)} samples ({int(train.labelled.sum())} yours, {int((train.who == FAST).sum())} yours above "
        f"{MAX_LABEL_KPH:.0f} km/h, {int((train.who == OPENPILOT).sum())} openpilot), "
        f"val {len(val)} ({int(val.labelled.sum())} yours, {int((val.who == OPENPILOT).sum())} openpilot, drives {sorted(map(str, val_drives))}), "
        f"held out {len(held & set(segs))} segments, from {len(replay_dirs)} replay dirs", flush=True)

  scale = train.labels[train.labelled].abs().mean(dim=(0, 1)).clamp(min=1e-3)
  print("label scale (mean |y| m, |yaw| rad, |yaw rate| rad/s):", np.round(scale.numpy(), 4), flush=True)
  original = load_policy(model_dir).to(device).eval().requires_grad_(False)
  model = load_policy(model_dir).to(device)
  base = evaluate(original, val, device)
  base_op = float((torch.from_numpy(base[OPENPILOT]) / scale).sum())
  for kind, name in ((YOURS, "yours"), (OPENPILOT, "openpilot's")):
    print(f"original on val, {name}: mean |error| y {base[kind][0]:.3f} m, yaw {base[kind][1]:.4f} rad, yaw rate {base[kind][2]:.4f} rad/s",
          flush=True)

  opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
  sc = scale.to(device)
  best, best_epoch = None, -1
  for epoch in range(EPOCHS):
    model.train()
    perm = torch.randperm(len(train))
    tot_fit = tot_anchor = 0.0
    for start in range(0, len(train), BATCH):
      rows = perm[start:start + BATCH]
      feats, labels, mask = train.batch(rows, device)
      out = run(model, feats)
      with torch.no_grad():
        ref = run(original, feats)
      weight = ANCHOR + ANCHOR_OP * ((mask == OPENPILOT) | (mask == FAST)).float()
      anchor = ((out - ref).abs().mean(dim=1) * weight).mean()
      mask = mask == YOURS
      fit = ((lateral(out)[mask] - labels[mask]).abs() / sc).mean() if mask.any() else torch.zeros((), device=device)
      loss = fit + ANCHOR * anchor
      opt.zero_grad()
      loss.backward()
      opt.step()
      tot_fit += float(fit.detach()) * len(rows); tot_anchor += float(anchor.detach()) * len(rows)
    v = evaluate(model, val, device)
    yours, op = (float((torch.from_numpy(v[k]) / scale).sum()) for k in (YOURS, OPENPILOT))
    score = yours + OP_PENALTY * max(0.0, op - base_op)
    print(f"epoch {epoch}: train fit {tot_fit / len(train):.3f} anchor {tot_anchor / len(train):.4f} | val yours y {v[YOURS][0]:.3f} m, "
          f"yaw {v[YOURS][1]:.4f}, yaw rate {v[YOURS][2]:.4f} | openpilot's y {v[OPENPILOT][0]:.3f} m, yaw {v[OPENPILOT][1]:.4f}, "
          f"yaw rate {v[OPENPILOT][2]:.4f} | score {score:.3f}", flush=True)
    if best is None or score < best:
      best, best_epoch = score, epoch
      torch.save(model.state_dict(), os.path.join(out_dir, "policy_finetuned.pt"))
  print(f"best epoch {best_epoch}", flush=True)

  model.load_state_dict(torch.load(os.path.join(out_dir, "policy_finetuned.pt")))
  model.eval()
  for tag, net in (("torch_original", original), ("finetuned", model)):
    os.makedirs(os.path.join(out_dir, tag), exist_ok=True)
  for s in sorted(held & set(segs)):
    r = dict(np.load(pairs[s][1]))  # the first replay dir: the chain compare_models reads
    hidden = np.concatenate([np.zeros((HISTORY - 1, FEATURES), np.float32), r["hidden"].astype(np.float32)])
    feats = torch.from_numpy(np.stack([hidden[k:k + HISTORY] for k in range(len(r["t_eof"]))])).to(device)
    for tag, net in (("torch_original", original), ("finetuned", model)):
      with torch.no_grad():
        out = run(net, feats).cpu().numpy()
      plan = out[:, :PLAN_N * PLAN_W].reshape(-1, PLAN_N, PLAN_W)
      np.savez_compressed(os.path.join(out_dir, tag, s + ".npz"), **{**r, "plan": plan.astype(np.float32)})
  print("held-out replays written", flush=True)


if __name__ == "__main__":
  main(*sys.argv[1:5])
