"""Run the openpilot.distill localizer on every segment under $LOCROOT/data (default ~/.comma/locroot) that has no output yet.

Segments whose GNSS products are not published yet fail; they are listed and can be retried in a few days.
Run with PYTHONPATH including openpilot.distill and the openpilot tree.
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(os.environ.get("LOCROOT", Path.home() / ".comma/locroot"))


def localize(segment):
  if (ROOT / "data" / segment / "localizer.safetensors").exists():
    return segment, "done before"
  import localizer.run as r
  r.PROJECT_ROOT = ROOT
  sys.argv = ["run", "--segment", segment]
  devnull = open(os.devnull, "w")
  saved = sys.stdout
  try:
    sys.stdout = devnull
    r.main()
    return segment, "ok"
  except Exception as e:  # noqa: BLE001
    return segment, f"failed: {type(e).__name__}"
  finally:
    sys.stdout = saved


def main():
  segments = sorted(os.listdir(ROOT / "data"))
  with ProcessPoolExecutor(6) as ex:
    results = list(ex.map(localize, segments, chunksize=2))
  failed = [s for s, r in results if r.startswith("failed")]
  print(f"localized {len(results) - len(failed)} of {len(results)}; failed {len(failed)}")
  for s in failed:
    print("  failed", s)


if __name__ == "__main__":
  main()
