"""Run an openpilot driving model (the stock one or the big chestnut one) over a recorded segment exactly as
modeld feeds it, frame by frame.

For every road camera frame: warp the narrow and wide frames into the model views with the device's own
calibration (get_warp_matrix, nearest-neighbour sampling of the Y, U and V planes, as frame_prepare does),
pack them into 6 channels, and keep modeld's queues: images 4 frames apart, the hidden state of the last
frames sampled every 4th (buffer length and feature shape read from the model), desire max-pooled over 4.
Saves every frame's parsed plan, lane lines, road edges and hidden state, and compares them with the modelV2
the device logged for the same frames.

With --chain, only frames 0, 4, 8, ... are run. The model's queues sample every 4th frame, so each chain of
frames 4 apart only ever sees its own hidden states and images: the outputs on that chain are exactly what
the 20 Hz loop gives, at a quarter of the compute.

Usage: python replay_model.py <segment dir> <model onnx> <out.npz> [max frames] [--chain]
"""
import base64, os, pickle, subprocess, sys
from bisect import bisect_right

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.common.transformations.model import get_warp_matrix
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.modeld.parse_model_outputs import Parser
from openpilot.tools.lib.logreader import LogReader

W, H = 512, 256
FRAME_SKIP = 4
DT_MDL = 0.05
UV_SCALE = np.array([[1.0, 1.0, 0.5], [1.0, 1.0, 0.5], [2.0, 2.0, 1.0]])


def read_log(path):
  narrow, wide, calib, delay, logged = [], {}, [], [], {}
  device = sensor = None
  for m in LogReader(path):
    w = m.which()
    if w == "narrowRoadEncodeIdx":
      e = m.narrowRoadEncodeIdx; narrow.append((e.frameId, e.segmentId))
    elif w == "wideRoadEncodeIdx":
      e = m.wideRoadEncodeIdx; wide[e.frameId] = e.segmentId
    elif w == "extrinsicsCalibration":
      calib.append((m.logMonoTime, np.array(m.extrinsicsCalibration.rpyCalib, dtype=np.float32)))
    elif w == "lateralDelay":
      delay.append((m.logMonoTime, m.lateralDelay.lateralDelay))
    elif w == "deviceState" and device is None:
      device = str(m.deviceState.deviceType)
    elif w == "narrowRoadCameraState" and sensor is None:
      sensor = str(m.narrowRoadCameraState.sensor)
    elif w == "modelV2":
      md = m.modelV2
      logged[md.frameId] = (np.array([le.y for le in md.laneLines]), np.array([re.y for re in md.roadEdges]),
                            np.array(md.position.y), m.logMonoTime)
  return narrow, wide, calib, delay, logged, (device, sensor)


def frames(path, w, h):
  proc = subprocess.Popen(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"],
                          stdout=subprocess.PIPE)
  size = w * h * 3 // 2
  while True:
    buf = proc.stdout.read(size)
    if len(buf) < size:
      break
    yield np.frombuffer(buf, dtype=np.uint8)
  proc.wait()


def prepare(yuv, w, h, m_inv):
  y = yuv[:w * h].reshape(h, w)
  u = yuv[w * h:w * h * 5 // 4].reshape(h // 2, w // 2)
  v = yuv[w * h * 5 // 4:].reshape(h // 2, w // 2)
  flags = cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP
  my = cv2.warpPerspective(y, m_inv, (W, H), flags=flags, borderMode=cv2.BORDER_REPLICATE)
  muv = m_inv * UV_SCALE
  mu = cv2.warpPerspective(u, muv, (W // 2, H // 2), flags=flags, borderMode=cv2.BORDER_REPLICATE)
  mv = cv2.warpPerspective(v, muv, (W // 2, H // 2), flags=flags, borderMode=cv2.BORDER_REPLICATE)
  return np.stack([my[0::2, 0::2], my[1::2, 0::2], my[0::2, 1::2], my[1::2, 1::2], mu, mv])


def at(timeline, t, default):
  i = bisect_right([x[0] for x in timeline], t) - 1
  return timeline[i][1] if i >= 0 else (timeline[0][1] if timeline else default)


def main(seg, onnx_path, out_path, max_frames=0, chain=False):
  max_frames = int(max_frames)
  step = FRAME_SKIP if chain else 1   # queue pushes per model run: one per frame, or one per chain step
  narrow, wide, calib, delay, logged, (device, sensor) = read_log(f"{seg}/rlog.zst")
  cams = DEVICE_CAMERAS[(device, sensor)]
  meta = {p.key: p.value for p in onnx.load(onnx_path, load_external_data=False).metadata_props}
  slices = pickle.loads(base64.b64decode(meta["output_slices"]))
  so = ort.SessionOptions()
  so.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "0"))   # 0 = all cores; set when running several at once
  sess = ort.InferenceSession(onnx_path, sess_options=so, providers=[p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                                                    if p in ort.get_available_providers()])
  ins = {i.name: i for i in sess.get_inputs()}
  dtype = {n: (np.uint8 if i.type == "tensor(uint8)" else np.float16 if i.type == "tensor(float16)" else np.float32)
           for n, i in ins.items()}
  fb_shape = tuple(ins["features_buffer"].shape)          # (1, entries, *feature)
  dp_shape = tuple(ins["desire_pulse"].shape)             # (1, entries, 8)
  feat_dim = int(np.prod(fb_shape[2:]))
  parser = Parser()
  cw, ch = cams.narrow_road.width, cams.narrow_road.height
  img_q = np.zeros((FRAME_SKIP // step + 1, 6, H // 2, W // 2), np.uint8)
  big_q = np.zeros_like(img_q)
  feat_q = np.zeros((FRAME_SKIP // step * fb_shape[1], feat_dim), np.float32)
  desire_q = np.zeros((FRAME_SKIP // step * dp_shape[1], dp_shape[2]), np.float32)
  prev_hidden = np.zeros(feat_dim, np.float32)
  by_index = {sid: fid for fid, sid in narrow}
  fid_list = [by_index.get(i) for i in range(len(narrow))]
  hid_rows, kept, plans, lanes, edges, probs, t_eof = [], [], [], [], [], [], []
  eof = {}
  for m in LogReader(f"{seg}/rlog.zst"):
    if m.which() == "narrowRoadEncodeIdx":
      eof[m.narrowRoadEncodeIdx.frameId] = m.narrowRoadEncodeIdx.timestampEof
  diffs = {"lane_lines": [], "road_edges": [], "plan_y": []}
  t_mono = {fid: lg[3] for fid, lg in logged.items()}
  for i, (fy, wy) in enumerate(zip(frames(f"{seg}/fcamera.hevc", cw, ch), frames(f"{seg}/ecamera.hevc", cw, ch), strict=False)):
    if max_frames and i >= max_frames:
      break
    if i % step:
      continue
    fid = fid_list[i] if i < len(fid_list) else None
    t = t_mono.get(fid, calib[0][0] if calib else 0)
    rpy = at(calib, t, np.zeros(3, np.float32))
    m_main = get_warp_matrix(rpy, cams.narrow_road.intrinsics, False)
    m_big = get_warp_matrix(rpy, cams.wide_road.intrinsics, True)
    img_q = np.concatenate([img_q[1:], prepare(fy, cw, ch, m_main)[None]])
    big_q = np.concatenate([big_q[1:], prepare(wy, cw, ch, m_big)[None]])
    feat_q = np.concatenate([feat_q[1:], prev_hidden[None]])
    desire_q = np.concatenate([desire_q[1:], np.zeros((1, dp_shape[2]), np.float32)])
    lat_delay = at(delay, t, 0.2)
    feed = {
      "img": img_q[::FRAME_SKIP // step].reshape(1, 12, H // 2, W // 2),
      "big_img": big_q[::FRAME_SKIP // step].reshape(1, 12, H // 2, W // 2),
      "features_buffer": feat_q[::FRAME_SKIP // step].reshape(fb_shape),
      "desire_pulse": desire_q.reshape(dp_shape[1], FRAME_SKIP // step, dp_shape[2]).max(1)[None],
      "traffic_convention": np.array([[1, 0]]),
      "action_t": np.array([[lat_delay + DT_MDL * 1.5, 0.3 + DT_MDL * 1.5]]),
    }
    feed = {k: v.astype(dtype[k]) for k, v in feed.items()}
    out = sess.run(None, feed)[0].astype(np.float32)[0]
    prev_hidden = out[slices["hidden_state"]]
    p = parser.parse_outputs({k: out[None, s] for k, s in slices.items()})
    hid_rows.append(prev_hidden.astype(np.float16)); kept.append(fid if fid is not None else -1)
    plans.append(p["plan"][0].astype(np.float32)); lanes.append(p["lane_lines"][0, :, :, 0].astype(np.float32))
    edges.append(p["road_edges"][0, :, :, 0].astype(np.float32)); probs.append(p["lane_lines_prob"][0].astype(np.float32))
    t_eof.append(eof.get(fid, 0))
    if fid in logged and i >= 100 and len(diffs["plan_y"]) < 10 ** 9:
      ll, re, py, _ = logged[fid]
      diffs["lane_lines"].append(np.abs(p["lane_lines"][0, :, :12, 0] - ll[:, :12]).mean(axis=1))
      diffs["road_edges"].append(np.abs(p["road_edges"][0, :, :12, 0] - re[:, :12]).mean(axis=1))
      diffs["plan_y"].append(abs(p["plan"][0, 10, 1] - py[10]))
  np.savez_compressed(out_path, hidden=np.array(hid_rows), frame_ids=np.array(kept), plan=np.array(plans),
                      lane_lines=np.array(lanes), road_edges=np.array(edges), lane_line_probs=np.array(probs),
                      t_eof=np.array(t_eof, dtype=np.int64))
  print(f"{len(plans)} frames; compared {len(diffs['plan_y'])} with the device's modelV2 (first 5 s skipped)")
  ll, re = np.array(diffs["lane_lines"]), np.array(diffs["road_edges"])
  print("lane lines mean |dy| over the first 12 points, median per line:", np.round(np.median(ll, axis=0), 2))
  print("road edges mean |dy| over the first 12 points, median per edge:", np.round(np.median(re, axis=0), 2))
  print(f"plan lateral position at {ModelConstants.T_IDXS[10]:.2f} s, median |dy|: {np.median(diffs['plan_y']):.2f} m")


if __name__ == "__main__":
  args = [a for a in sys.argv[1:] if a != "--chain"]
  main(*args[:4], chain="--chain" in sys.argv)
