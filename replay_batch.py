"""Replay a driving model over many segments in chain mode, a few segments at a time.

Usage: python replay_batch.py <model onnx> <out dir> <workers> <segment dir> [<segment dir> ...]
Segments already replayed into <out dir> are skipped.
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def run(args):
  seg, onnx_path, out_dir = args
  import contextlib, io
  import replay_model
  out = os.path.join(out_dir, os.path.basename(seg.rstrip("/")) + ".npz")
  if os.path.exists(out):
    return seg, "done before"
  try:
    with contextlib.redirect_stdout(io.StringIO()):
      replay_model.main(seg, onnx_path, out, 0, chain=True)
    return seg, "ok"
  except Exception as e:  # noqa: BLE001
    return seg, f"failed: {type(e).__name__}: {e}"


def main():
  onnx_path, out_dir, workers, *segments = sys.argv[1:]
  os.makedirs(out_dir, exist_ok=True)
  with ProcessPoolExecutor(int(workers)) as ex:
    for seg, status in ex.map(run, [(s, onnx_path, out_dir) for s in segments]):
      print(os.path.basename(seg.rstrip("/")), status, flush=True)


if __name__ == "__main__":
  main()
