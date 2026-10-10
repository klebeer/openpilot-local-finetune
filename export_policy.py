"""Write a fine-tuned policy back into the original policy ONNX: the same graph, inputs, outputs and metadata, with each
trained weight in place of the original one, in the original's precision.

Each ONNX weight is matched to the converted module's tensor holding the same values (as is or transposed, as
onnx2torch stores Gemm weights), so the mapping needs no knowledge of names. The written file is then run with
onnxruntime against the fine-tuned torch module on random features.

Usage: python export_policy.py <model dir with driving_policy.onnx> <policy_finetuned.pt> <out onnx>
"""
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import numpy_helper

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_policy import HISTORY, load_policy, run  # noqa: E402


def match(original: dict[str, np.ndarray], weight: np.ndarray) -> list[tuple[str, bool]]:
  """Names (and whether transposed) of the module's tensors equal to this ONNX weight."""
  out = []
  for name, value in original.items():
    if value.shape == weight.shape and np.array_equal(value, weight):
      out.append((name, False))
    elif value.ndim == 2 and value.T.shape == weight.shape and np.array_equal(value.T, weight):
      out.append((name, True))
  return out


def main(model_dir, state_path, out_path):
  src = os.path.join(model_dir, "driving_policy.onnx")
  graph = onnx.load(src)
  original = {k: v.detach().numpy() for k, v in load_policy(model_dir).state_dict().items() if v.is_floating_point()}
  tuned_module = load_policy(model_dir)
  tuned_module.load_state_dict(torch.load(state_path, map_location="cpu"))
  tuned = {k: v.detach().numpy() for k, v in tuned_module.state_dict().items()}
  used = set()
  for init in graph.graph.initializer:
    weight = numpy_helper.to_array(init)
    if not np.issubdtype(weight.dtype, np.floating):
      continue
    found = [m for m in match(original, weight.astype(np.float32)) if m[0] not in used]
    if not found:
      sys.exit(f"{init.name}: no tensor in the module holds it")
    name, transposed = found[0]
    used.add(name)
    new = tuned[name].T if transposed else tuned[name]
    init.CopyFrom(numpy_helper.from_array(new.astype(weight.dtype), init.name))
  missing = set(original) - used
  if missing:
    print("module tensors not written back (constants, not weights):", sorted(missing))
  onnx.save(graph, out_path)

  rng = np.random.default_rng(0)
  feats = (rng.standard_normal((4, HISTORY, 512)) * 0.5).astype(np.float32)
  sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
  ref = np.stack([sess.run(None, {"desire_pulse": np.zeros((1, HISTORY, 8), np.float16), "traffic_convention": np.array([[1, 0]], np.float16),
                                  "features_buffer": f[None].astype(np.float16)})[0][0] for f in feats]).astype(np.float32)
  with torch.no_grad():
    got = run(tuned_module.eval(), torch.from_numpy(feats)).numpy()
  print(f"wrote {out_path}; onnxruntime vs torch, max |diff| {np.abs(ref - got).max():.3f}, plan y at 1 s {np.round(ref[:, 151], 3)} vs "
        f"{np.round(got[:, 151], 3)}")


if __name__ == "__main__":
  main(*sys.argv[1:4])
