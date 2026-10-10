# openpilot-local-finetune

Fine-tune an openpilot driving model on your own drives, on a laptop. The scripts replay comma's driving models over
recorded segments exactly as `modeld` feeds them, build the path the car really took from a GNSS localizer, adjust
the policy (the part that plans the path) with the vision model frozen, and compare models on held-out segments.

Nothing here touches a car. A model that passes the comparison still has to be compiled for the device and checked
there before anyone drives with it.

## What you need

| | |
|---|---|
| Python 3.12 | `onnx`, `onnxruntime`, `onnx2torch`, `torch` (Apple MPS or CUDA), `opencv-python`, `numpy`, `scipy`, `safetensors` |
| `ffmpeg` | decodes `fcamera.hevc` and `ecamera.hevc` |
| an openpilot tree on `PYTHONPATH` | sunnypilot for `camera_offset_helper`; [zoompilot](https://github.com/klebeer/zoompilot/tree/cx5-2025-non-mrcc) for `context_offset.py` (optional, replays that shift) and `mapd/map_ways.py` with offline map tiles (street types in `compare_models.py`) |
| [openpilot.distill](https://github.com/YassineYousfi/openpilot.distill) on `PYTHONPATH` | its localizer, for `localize_all.py` |
| your segments | under `$LOCROOT/data/<segment>/` (default `~/.comma/locroot`): `rlog.zst`, `fcamera.hevc`, `ecamera.hevc`, then `frame_info.safetensors` and `localizer.safetensors` from the localizer |

No model weights or drive data are in this repository.

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1 | `fetch_model.py <openpilot commit> <dir>` | downloads a model's ONNX files from comma's Git LFS and checks their hashes |
| 2 | `localize_all.py` | runs the localizer on every segment that has no output yet |
| 3 | `replay_split.py` / `replay_split_loop.sh` | replays a split model (`driving_vision.onnx` + `driving_policy.onnx`) over a segment, applying the camera offset the device applied, and compares with the logged `modelV2`; `--chain` runs every 4th frame, `--chains=1,2,3` the other three in one pass |
| 3' | `replay_model.py` / `replay_loop.sh` / `replay_batch.py` | the same for single-file models (`driving_supercombo.onnx`) |
| 4 | `stock_to_torch.py` | converts an ONNX model to PyTorch with the converters `onnx2torch` lacks |
| 5 | `train_policy.py` | fine-tunes the policy on the frames you steered, anchored to the original everywhere else |
| 6 | `compare_models.py` | lateral error of each model's plan against the real path at 1 s and 2 s, split by who steered and by street type |

Every script documents its arguments in its docstring.

## How the fine-tune works

- **Inputs.** Each sample is what the policy sees on the device: the vision hidden states of the last 25 frames,
  4 frames apart, from the replay. The vision model is not trained, so it runs once per segment.
- **Labels.** The labels follow `openpilot.distill`'s plan targets, in the device's own calibrated frame: lateral
  position, heading change and yaw rate up to 3.2 s. openpilot steers from the plan's heading and yaw rate, so all
  three are fitted.
- **Which frames teach.** Only frames where you steered through the next 3 s. Frames where openpilot steered would
  teach the model to repeat itself. Those frames only anchor the outputs to the original policy's, with extra weight
  so its own driving does not change.
- **Which epoch is kept.** The one closest to your path on a validation set of whole drives, penalised for any loss
  of accuracy where openpilot steered.

## Known limits

- **Desire.** Desire is zero in the replay, so turns and lane changes the device made with a desire are replayed
  without it.
- **Traffic side.** The traffic convention is fixed to a left-hand-drive car.
- **Camera offset.** It is replayed from the logged outputs of the model the device ran then, which may not be the
  model being replayed.

## License

MIT. See `LICENSE`.
