"""Convert openpilot's stock driving_supercombo.onnx (opset 20) to a PyTorch module with onnx2torch.

onnx2torch lacks converters for six operators at the versions this model uses: Cast and Identity keep an
older version's semantics, Reshape needs allowzero, ReduceMean and ReduceL2 at 18 take axes as an input.
"""
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx2torch import convert
from onnx2torch.node_converters.registry import _CONVERTER_REGISTRY, add_converter
from onnx2torch.utils.common import OperationConverterResult, onnx_mapping_from_node


def reuse(op, new_version):
  old = max((k for k in _CONVERTER_REGISTRY if k.operation_type == op and k.domain == "" and k.version < new_version),
            key=lambda k: k.version)
  add_converter(operation_type=op, version=new_version)(_CONVERTER_REGISTRY[old])


for op, v in (("Cast", 19), ("Identity", 19)):
  reuse(op, v)


class Gelu(torch.nn.Module):
  def __init__(self, approximate):
    super().__init__()
    self.approximate = approximate

  def forward(self, x):
    return torch.nn.functional.gelu(x, approximate=self.approximate)


@add_converter(operation_type="Gelu", version=20)
def _gelu(node, graph):
  approx = node.attributes.get("approximate", "none")
  return OperationConverterResult(torch_module=Gelu(approx.decode() if isinstance(approx, bytes) else approx),
                                  onnx_mapping=onnx_mapping_from_node(node))


class Reshape(torch.nn.Module):
  def __init__(self, allowzero):
    super().__init__()
    self.allowzero = allowzero

  def forward(self, x, shape):
    dims = [int(d) for d in shape.tolist()]
    if not self.allowzero:
      dims = [x.shape[i] if d == 0 else d for i, d in enumerate(dims)]
    return torch.reshape(x, dims)


@add_converter(operation_type="Reshape", version=19)
def _reshape(node, graph):
  return OperationConverterResult(torch_module=Reshape(bool(node.attributes.get("allowzero", 0))),
                                  onnx_mapping=onnx_mapping_from_node(node))


class Reduce(torch.nn.Module):
  def __init__(self, kind, keepdims, axes):
    super().__init__()
    self.kind, self.keepdims, self.axes = kind, keepdims, axes

  def forward(self, x, axes=None):
    dims = self.axes if axes is None else tuple(int(a) for a in axes.tolist())
    if self.kind == "mean":
      return torch.mean(x, dim=dims, keepdim=self.keepdims)
    return torch.linalg.vector_norm(x, ord=2, dim=dims, keepdim=self.keepdims)


def _reduce(kind):
  def conv(node, graph):
    keepdims = bool(node.attributes.get("keepdims", 1))
    axes = None
    if len(node.input_values) > 1 and node.input_values[1] in graph.initializers:
      axes = tuple(int(a) for a in graph.initializers[node.input_values[1]].to_torch().tolist())
    return OperationConverterResult(torch_module=Reduce(kind, keepdims, axes), onnx_mapping=onnx_mapping_from_node(node))
  return conv


add_converter(operation_type="ReduceMean", version=18)(_reduce("mean"))
add_converter(operation_type="ReduceL2", version=18)(_reduce("l2"))


def main(path, out):
  tm = convert(onnx.load(path))
  print(f"converted: {sum(p.numel() for p in tm.parameters()) / 1e6:.1f}M trainable parameters, "
        f"{sum(b.numel() for b in tm.buffers()) / 1e6:.2f}M in buffers")
  s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
  rng = np.random.default_rng(0)
  dtypes = {"tensor(uint8)": np.uint8, "tensor(float16)": np.float16, "tensor(float)": np.float32}
  feed = {i.name: (rng.integers(0, 256, size=i.shape, dtype=np.uint8) if i.type == "tensor(uint8)"
                   else (rng.standard_normal(i.shape) * 0.1).astype(dtypes[i.type])) for i in s.get_inputs()}
  ref = s.run(None, feed)[0]
  with torch.no_grad():
    got = tm(*[torch.from_numpy(feed[i.name]) for i in s.get_inputs()])
  got = (got[0] if isinstance(got, (list, tuple)) else got).float().numpy()
  ref = ref.astype(np.float32)
  print(f"max abs diff vs onnxruntime: {np.abs(got - ref).max():.2e} (outputs up to {np.abs(ref).max():.1f})")
  torch.save(tm, out)


if __name__ == "__main__":
  main(sys.argv[1], sys.argv[2])
