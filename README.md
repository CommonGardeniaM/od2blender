# mmd-trace (MVP)

Monocular video to MMD VMD motion (MVP). This tool follows the 3-stage pipeline:

1. Pose extraction to JSON
2. Smoothing / gap filling
3. Retargeting to PMX bones and VMD export (center/groove included)

## Install

Use uv to install dependencies and run the CLI.

```bash
uv venv
uv pip install -r requirements.txt
uv pip install -e .
```

Download MediaPipe pose landmarker model:

```bash
python - <<'PY'
from pathlib import Path
import urllib.request

url = "https://storage.googleapis.com/mediapipe-assets/pose_landmarker.task"
target = Path("models") / "pose_landmarker.task"
target.parent.mkdir(parents=True, exist_ok=True)
urllib.request.urlretrieve(url, target)
print("Saved", target)
PY
```

## Usage

```bash
python -m mmd_trace trace --video input.mp4 --pmx model.pmx --out out_dir --config config.yaml --smooth
```

Disable smoothing:

```bash
python -m mmd_trace trace --video input.mp4 --pmx model.pmx --out out_dir --config config.yaml --no-smooth
```

Outputs in `out_dir`:

- `pose_raw.json`
- `pose_smooth.json`
- `motion.vmd`
- `debug_overlay.mp4` (optional)

## Config Example

See `config_example.yaml` for all settings. Important notes:

- Bone names must match the PMX model.
- Axis conversion is applied via `axis_map`.
- Center/groove pattern can be `A` (default) or `B`.

## License and Usage Notes

This project references the ideas and workflow from:

- https://github.com/miu200521358/mmd-auto-trace-4
- https://github.com/miu200521358/mmd-auto-trace-4/wiki/01.-%E6%A6%82%E8%A6%81
- https://github.com/miu200521358/mmd-auto-trace-4/wiki/02.%E4%BD%BF%E7%94%A8%E6%8A%80%E8%A1%93
- https://github.com/miu200521358/mmd-auto-trace-4/wiki/03.%E5%88%A9%E7%94%A8%E8%A6%8F%E7%B4%84

Please review their usage policy and ensure that generated outputs comply with the terms.

## Notes

- Only MediaPipe provider is included in MVP.
- IK / fingers / facial morphs are out of scope for MVP.
