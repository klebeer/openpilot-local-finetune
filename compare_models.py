"""Compare driving models' planned paths with the path the car really took, on held-out segments.

Inputs per segment: replay_model.py outputs (chain mode is enough) for each model, the localizer output
(frame_info/localizer.safetensors in the segment dir) and the rlog. Ground truth is the localized future
position at about 1 s and 2 s, expressed in the frame the model sees: the device's own calibration
(extrinsicsCalibration), not the localizer's. Each sample is labelled by who steered through the next 3 s,
by the street from the offline map (map_ways: two-way or one-way), and by whether the model saw clear ego
lane lines. Reports lateral error (median / p90) and, on two-way streets, the signed offset model minus
driver (positive: the model would sit further right than the driver did).

Usage: python compare_models.py <name>=<replay dir> [<name>=<replay dir> ...] -- <segment dir> [...]
"""
import os
import sys
from bisect import bisect_right
from collections import defaultdict

import numpy as np
from safetensors.numpy import load
from openpilot.common.transformations.coordinates import ecef2geodetic
from openpilot.common.transformations.orientation import rot_from_euler, rot_from_quat
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.sunnypilot.mapd.map_ways import MapWays, bearing_deg
from openpilot.tools.lib.logreader import LogReader

T_IDXS = np.asarray(ModelConstants.T_IDXS)
HORIZONS = (int(np.argmin(np.abs(T_IDXS - 1.0))), int(np.argmin(np.abs(T_IDXS - 2.0))))
WHO_WINDOW_S = 3.0
CLEAR_LINES = 0.5
FAST_KPH = 80.0


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


def future_lateral(states, frame_t, fidx, calib_rpy):
  """Driver's lateral position at the HORIZONS, in the device calibrated frame (y positive to the right)."""
  query = T_IDXS[list(HORIZONS)] + frame_t[fidx]
  i = np.clip(np.searchsorted(frame_t, query) - 1, 0, len(frame_t) - 2)
  w = ((query - frame_t[i]) / (frame_t[i + 1] - frame_t[i]))[:, None]
  pos = states[i, 0:3] * (1 - w) + states[i + 1, 0:3] * w
  device_from_ecef = rot_from_quat(states[fidx, 3:7]).T
  calib_from_device = rot_from_euler(calib_rpy).T
  return (calib_from_device @ device_from_ecef @ (pos - states[fidx, 0:3]).T).T[:, 1]


def main():
  sep = sys.argv.index("--")
  models = dict(a.split("=", 1) for a in sys.argv[1:sep])
  segments = sys.argv[sep + 1:]
  ways = MapWays()
  err = defaultdict(list)       # (model, group) -> [abs error at horizons]
  bias = defaultdict(list)      # model -> signed offset at 2 s on two-way streets, driver steering
  fast_bias = defaultdict(list)  # model -> signed offset at 2 s at FAST_KPH or faster, openpilot steering
  for seg in segments:
    name = os.path.basename(seg.rstrip("/"))
    if not os.path.exists(f"{seg}/localizer.safetensors"):
      continue
    loc = load(open(f"{seg}/localizer.safetensors", "rb").read())
    info = load(open(f"{seg}/frame_info.safetensors", "rb").read())
    frame_t, states = info["fcamera/t"], loc["frame_states"]
    calib, lat = log_timelines(f"{seg}/rlog.zst")
    runs = {m: np.load(os.path.join(d, name + ".npz")) for m, d in models.items() if os.path.exists(os.path.join(d, name + ".npz"))}
    if len(runs) != len(models):
      continue
    ref = next(iter(runs.values()))
    for k, t_eof in enumerate(ref["t_eof"]):
      t = t_eof * 1e-9 - 0.008
      fidx = int(np.argmin(np.abs(frame_t - t)))
      if abs(frame_t[fidx] - t) > 0.02 or fidx + 50 >= len(frame_t):
        continue
      v = float(np.linalg.norm(states[fidx, 7:10]))
      if v < 3.0:
        continue
      win = [x for x in lat if t <= x[0] <= t + WHO_WINDOW_S]
      if not win:
        continue
      who = "driver" if not any(x[1] for x in win) else "openpilot" if all(x[1] for x in win) else None
      if who is None:
        continue
      truth = future_lateral(states, frame_t, fidx, at(calib, t))
      lat_deg, lon_deg, _ = ecef2geodetic(states[fidx, 0:3])
      la2, lo2, _ = ecef2geodetic(states[min(fidx + 20, len(states) - 1), 0:3])
      way = ways.match(lat_deg, lon_deg, bearing_deg((lat_deg, lon_deg), (la2, lo2)))
      two_way = way is not None and not way.one_way
      clear = bool(ref["lane_line_probs"][k][1] >= CLEAR_LINES and ref["lane_line_probs"][k][2] >= CLEAR_LINES)
      for m, r in runs.items():
        pred = r["plan"][k][list(HORIZONS), 1]
        e = np.abs(pred - truth)
        err[(m, who, "all")].append(e)
        if v * 3.6 >= FAST_KPH:
          err[(m, who, f"{FAST_KPH:.0f}+ km/h")].append(e)
          if who == "openpilot":
            fast_bias[m].append(pred[1] - truth[1])
        if two_way:
          err[(m, who, "two-way" + ("" if clear else ", no clear lines"))].append(e)
          if who == "driver":
            bias[m].append(pred[1] - truth[1])
  print(f"lateral error vs the real path, m: median / p90 at {T_IDXS[HORIZONS[0]]:.1f} s and {T_IDXS[HORIZONS[1]]:.1f} s")
  for group in sorted({(w, g) for _, w, g in err}):
    print(f"\n  {group[0]}, {group[1]}")
    for m in models:
      e = err.get((m, *group))
      if not e:
        continue
      e = np.array(e)
      print(f"    {m:10} n={len(e):5d}  {np.median(e[:, 0]):.2f} / {np.percentile(e[:, 0], 90):.2f}   "
            f"{np.median(e[:, 1]):.2f} / {np.percentile(e[:, 1], 90):.2f}")
  print("\nsigned offset at 2 s on two-way streets while the driver steered (positive: model further right):")
  for m in models:
    if bias[m]:
      b = np.array(bias[m])
      print(f"    {m:10} n={len(b):5d}  median {np.median(b):+.2f} m, mean {b.mean():+.2f} m")
  print(f"\nsigned offset at 2 s at {FAST_KPH:.0f}+ km/h while openpilot steered (positive: model further right than the car went):")
  for m in models:
    if fast_bias[m]:
      b = np.array(fast_bias[m])
      print(f"    {m:10} n={len(b):5d}  median {np.median(b):+.2f} m, mean {b.mean():+.2f} m")


if __name__ == "__main__":
  main()
