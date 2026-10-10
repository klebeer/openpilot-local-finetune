"""Download a driving model from openpilot's history: the ONNX files at a commaai/openpilot commit, fetched from comma's
Git LFS server and checked against each pointer's sha256 and size.

sunnypilot's model list gives each bundle's openpilot commit as `ref` (North Nevada V2:
83b81b83cdf76a577946a1567f4644e1a018443c). Split models ship driving_vision.onnx and driving_policy.onnx; older ones a
single driving_supercombo.onnx.

Usage: python fetch_model.py <openpilot commit> <out dir> [file ...]   (default: driving_vision.onnx driving_policy.onnx)
"""
import hashlib
import json
import os
import sys
import urllib.request

RAW = "https://raw.githubusercontent.com/commaai/openpilot/{ref}/selfdrive/modeld/models/{name}"
LFS = "https://gitlab.com/commaai/openpilot-lfs.git/info/lfs/objects/batch"


def pointer(ref: str, name: str) -> tuple[str, int]:
  text = urllib.request.urlopen(RAW.format(ref=ref, name=name), timeout=60).read().decode()
  fields = dict(line.split(" ", 1) for line in text.splitlines() if " " in line)
  return fields["oid"].removeprefix("sha256:"), int(fields["size"])


def main(ref: str, out_dir: str, *names: str) -> None:
  names = names or ("driving_vision.onnx", "driving_policy.onnx")
  os.makedirs(out_dir, exist_ok=True)
  wanted = {name: pointer(ref, name) for name in names}
  body = {"operation": "download", "transfers": ["basic"], "objects": [{"oid": oid, "size": size} for oid, size in wanted.values()]}
  req = urllib.request.Request(LFS, data=json.dumps(body).encode(),
                               headers={"Accept": "application/vnd.git-lfs+json", "Content-Type": "application/vnd.git-lfs+json"})
  hrefs = {o["oid"]: o["actions"]["download"]["href"] for o in json.load(urllib.request.urlopen(req, timeout=60))["objects"] if "actions" in o}
  for name, (oid, size) in wanted.items():
    data = urllib.request.urlopen(hrefs[oid], timeout=600).read()
    if hashlib.sha256(data).hexdigest() != oid or len(data) != size:
      sys.exit(f"{name}: does not match its LFS pointer")
    with open(os.path.join(out_dir, name), "wb") as f:
      f.write(data)
    print(f"{name}: {size} bytes, sha256 ok")


if __name__ == "__main__":
  main(*sys.argv[1:])
