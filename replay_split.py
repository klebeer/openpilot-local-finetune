"""Run a split openpilot driving model (driving_vision.onnx + driving_policy.onnx, as North Nevada V2 ships) over a
recorded segment as modeld feeds it, and compare with the modelV2 the device logged.

Frames, warp and image queue are replay_model.py's. The camera is moved as the device moved it: CameraOffset from
initData, ContextOffset (in trees that have it) replayed on the logged modelV2 in log order, and the transform's offset advanced only when a
new calibration arrives, as modeld does. The policy's features_buffer holds the vision hidden states 4 frames apart,
the current frame's last; desire stays zero. Saves the same fields as replay_model.py, so compare_models.py reads
either.

With --chain, only frames 0, 4, 8, ... are run: each chain only ever sees its own hidden states and images.
--chains=1,2,3 runs those other chains in the same pass over the video, each to <out> with {c} replaced by its offset.

Usage: python replay_split.py <segment dir> <model dir with driving_vision.onnx and driving_policy.onnx> <out.npz>
                              [max frames] [--chain | --chains=<offsets>]
"""
import base64
import os
import pickle
import sys

import numpy as np
import onnx
import onnxruntime as ort
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.common.transformations.model import get_warp_matrix
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.modeld.parse_model_outputs import Parser
from openpilot.sunnypilot.modeld_v2.camera_offset_helper import CameraOffsetHelper
from openpilot.tools.lib.logreader import LogReader

try:
  from openpilot.sunnypilot.modeld_v2.context_offset import ContextOffset
except ImportError:  # only in trees that shift the camera by context; elsewhere the offset is CameraOffset alone
  ContextOffset = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay_model import FRAME_SKIP, H, W, at, frames, prepare, read_log  # noqa: E402

def session(path):
  meta = {p.key: p.value for p in onnx.load(path, load_external_data=False).metadata_props}
  so = ort.SessionOptions()
  so.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "0"))
  sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
  ins = {i.name: i for i in sess.get_inputs()}
  dtype = {n: (np.uint8 if i.type == "tensor(uint8)" else np.float16 if i.type == "tensor(float16)" else np.float32) for n, i in ins.items()}
  return sess, ins, dtype, pickle.loads(base64.b64decode(meta["output_slices"]))


def fingerprint(path):
  return next((str(m.carParams.carFingerprint) for m in LogReader(path) if m.which() == "carParams"), "")


def applied_offsets(path):
  """The camera offset in the transform each frame was run with, by frameId."""
  ctx = ContextOffset(fingerprint(path), 0.05) if ContextOffset is not None else None
  base, actual, wanted, v, height = 0.0, 0.0, 0.0, 0.0, 1.22
  out, heights = {}, {}
  for m in LogReader(path):
    w = m.which()
    if w == "initData":
      entries = {e.key: e.value for e in m.initData.params.entries}
      base = float(entries.get("CameraOffset", b"0") or 0)
      wanted = ctx.camera_offset(base) if ctx is not None else base
    elif w == "carState":
      v = m.carState.vEgo
    elif w == "extrinsicsCalibration":
      actual = 0.9 * actual + 0.1 * wanted
      height = m.extrinsicsCalibration.height[0] if len(m.extrinsicsCalibration.height) else 1.22
    elif w == "modelV2":
      out[m.modelV2.frameId], heights[m.modelV2.frameId] = actual, height
      if ctx is not None:
        ctx.update(m.modelV2, max(v, 0.0))
        wanted = ctx.camera_offset(base)
  return out, heights


class Chain:
  """Queues and outputs of one run of the model: every frame (per=4), or one frame in 4 (per=1)."""

  def __init__(self, per, fb_shape, dp_shape):
    self.per, self.dp_shape = per, dp_shape
    self.img_q = np.zeros((per + 1, 6, H // 2, W // 2), np.uint8)
    self.big_q = np.zeros_like(self.img_q)
    self.feat_q = np.zeros((per * fb_shape[1], fb_shape[2]), np.float32)
    self.desire_q = np.zeros((per * dp_shape[1], dp_shape[2]), np.float32)
    self.rows = {k: [] for k in ("hidden", "frame_ids", "plan", "lane_lines", "road_edges", "lane_line_probs", "t_eof", "camera_offset")}

  def save(self, path):
    np.savez_compressed(path, **{k: np.array(v, dtype=np.int64 if k == "t_eof" else np.float32 if k == "camera_offset" else None)
                                 for k, v in self.rows.items()})


def main(seg, model_dir, out_path, max_frames=0, chain=False, chains=(0,)):
  """`chains`: the frame offsets (0-3) to run in chain mode, each written to out_path with {c} replaced."""
  max_frames = int(max_frames)
  chains = tuple(chains) if chain else (0,)
  narrow, wide, calib, delay, logged, (device, sensor) = read_log(f"{seg}/rlog.zst")
  offsets, heights = applied_offsets(f"{seg}/rlog.zst")
  cams = DEVICE_CAMERAS[(device, sensor)]
  vision, v_ins, v_dtype, v_slices = session(os.path.join(model_dir, "driving_vision.onnx"))
  policy, p_ins, p_dtype, p_slices = session(os.path.join(model_dir, "driving_policy.onnx"))
  fb_shape, dp_shape = tuple(p_ins["features_buffer"].shape), tuple(p_ins["desire_pulse"].shape)
  parser = Parser()
  cw, ch = cams.narrow_road.width, cams.narrow_road.height
  per = 1 if chain else FRAME_SKIP
  runs = {c: Chain(per, fb_shape, dp_shape) for c in chains}
  by_index = {sid: fid for fid, sid in narrow}
  fid_list = [by_index.get(i) for i in range(len(narrow))]
  t_mono = {fid: lg[3] for fid, lg in logged.items()}
  eof = {}
  for m in LogReader(f"{seg}/rlog.zst"):
    if m.which() == "narrowRoadEncodeIdx":
      eof[m.narrowRoadEncodeIdx.frameId] = m.narrowRoadEncodeIdx.timestampEof
  diffs = {"lane_lines": [], "road_edges": [], "plan_y": []}
  for i, (fy, wy) in enumerate(zip(frames(f"{seg}/fcamera.hevc", cw, ch), frames(f"{seg}/ecamera.hevc", cw, ch), strict=False)):
    if max_frames and i >= max_frames:
      break
    run = runs.get(i % FRAME_SKIP) if chain else runs[0]
    if run is None:
      continue
    fid = fid_list[i] if i < len(fid_list) else None
    t = t_mono.get(fid, calib[0][0] if calib else 0)
    rpy = at(calib, t, np.zeros(3, np.float32))
    off, height = offsets.get(fid, 0.0), heights.get(fid, 1.22)
    m_main = CameraOffsetHelper.apply_camera_offset(get_warp_matrix(rpy, cams.narrow_road.intrinsics, False), cams.narrow_road.intrinsics, height, off)
    m_big = CameraOffsetHelper.apply_camera_offset(get_warp_matrix(rpy, cams.wide_road.intrinsics, True), cams.wide_road.intrinsics, height, off)
    run.img_q = np.concatenate([run.img_q[1:], prepare(fy, cw, ch, m_main)[None]])
    run.big_q = np.concatenate([run.big_q[1:], prepare(wy, cw, ch, m_big)[None]])
    v_feed = {"img": run.img_q[::run.per].reshape(1, 12, H // 2, W // 2), "big_img": run.big_q[::run.per].reshape(1, 12, H // 2, W // 2)}
    v_out = vision.run(None, {k: v.astype(v_dtype[k]) for k, v in v_feed.items()})[0].astype(np.float32)[0]
    hidden = v_out[v_slices["hidden_state"]]
    run.feat_q = np.concatenate([run.feat_q[1:], hidden[None]])
    run.desire_q = np.concatenate([run.desire_q[1:], np.zeros((1, dp_shape[2]), np.float32)])
    p_feed = {"features_buffer": run.feat_q[run.per - 1::run.per][None],
              "desire_pulse": run.desire_q.reshape(dp_shape[1], run.per, dp_shape[2]).max(1)[None], "traffic_convention": np.array([[1, 0]])}
    p_out = policy.run(None, {k: v.astype(p_dtype[k]) for k, v in p_feed.items()})[0].astype(np.float32)[0]
    p = parser.parse_vision_outputs({k: v_out[None, s] for k, s in v_slices.items() if k != "pad"})
    p.update(parser.parse_policy_outputs({k: p_out[None, s] for k, s in p_slices.items() if k != "pad"}))
    for k, v in (("hidden", hidden.astype(np.float16)), ("frame_ids", fid if fid is not None else -1), ("plan", p["plan"][0]),
                 ("lane_lines", p["lane_lines"][0, :, :, 0]), ("road_edges", p["road_edges"][0, :, :, 0]),
                 ("lane_line_probs", p["lane_line_probs"][0] if "lane_line_probs" in p else p["lane_lines_prob"][0]),
                 ("t_eof", eof.get(fid, 0)), ("camera_offset", off)):
      run.rows[k].append(v)
    if fid in logged and i >= 100:
      ll, re, py, _ = logged[fid]
      diffs["lane_lines"].append(np.abs(p["lane_lines"][0, :, :12, 0] - ll[:, :12]).mean(axis=1))
      diffs["road_edges"].append(np.abs(p["road_edges"][0, :, :12, 0] - re[:, :12]).mean(axis=1))
      diffs["plan_y"].append(abs(p["plan"][0, 10, 1] - py[10]))
  for c, run in runs.items():
    run.save(out_path.replace("{c}", str(c)))
  print(f"{sum(len(r.rows['plan']) for r in runs.values())} frames; compared {len(diffs['plan_y'])} with the device's modelV2 (first 5 s skipped)")
  if diffs["plan_y"]:
    ll, re = np.array(diffs["lane_lines"]), np.array(diffs["road_edges"])
    print("lane lines mean |dy| over the first 12 points, median per line:", np.round(np.median(ll, axis=0), 2))
    print("road edges mean |dy| over the first 12 points, median per edge:", np.round(np.median(re, axis=0), 2))
    print(f"plan lateral position at {ModelConstants.T_IDXS[10]:.2f} s, median |dy|: {np.median(diffs['plan_y']):.2f} m")


if __name__ == "__main__":
  flags = [a for a in sys.argv[1:] if a.startswith("--")]
  args = [a for a in sys.argv[1:] if not a.startswith("--")]
  picked = next((f.split("=", 1)[1] for f in flags if f.startswith("--chains=")), None)
  main(*args[:4], chain="--chain" in flags or picked is not None, chains=tuple(int(c) for c in picked.split(",")) if picked else (0,))
