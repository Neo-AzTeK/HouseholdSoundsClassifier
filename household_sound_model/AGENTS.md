# Project guidance

This directory contains the household sound model and its training, evaluation, and inference code. Keep all new project files, downloaded data, model artifacts, and notes inside this directory. See `PLAN.md` for the full history of decisions.

The selected checkpoint is `artifacts/household_sound.pt`; the original unmixed model is preserved as `artifacts/household_sound_v1.pt`. `RUN_REPORT.md` records the dataset, commands, metrics, and limitations. Keep these checkpoints available when experimenting with retraining so there is always a usable baseline.
`TRAINING_EXPLAINED.md` explains the data, processing, optimization, validation, inference, and retraining workflow for the selected checkpoint.
`docs/Household Sound Model.md` is the illustrated Obsidian-compatible technical note. `docs/generate_figures.py` regenerates its plots from CC0 public clips and the actual preprocessing code; keep both figures and source attribution with the note.

## Fixed requirements

- Inputs are mono audio from one Nordic Thingy:53 microphone. The existing continuous stream is about 4,031 samples/s after filtering, 4:1 decimation, and 8-bit compression. Do not change the Thingy firmware for the first model.
- Outputs are one JSON object with five independent scores in `[0, 1]`: `voices`, `machines`, `collisions`, `steps`, `music`. Speech and music may both score high. Return scores every 0.5–1 second for live inference.
- Labels describe audible content. Speaker speech counts as voices; singing over music can count as both. Collisions include brief impacts such as dropped objects and door slams.
- Start with public training data only. Recordings from the target room may be used for testing. Preserve dataset source, clip ID, license, original labels, mapped labels, and split so the dataset can expand reproducibly.
- Aim for a first working model covering all five classes. Include both file and live inference entry points. The eventual target is Raspberry Pi 5 with AI HAT+ 2 (Hailo-10H), but a real, measured baseline comes first.
- Current computer has around 30 GB free and an RX 6800S; a second computer has an RTX 2060 Super. Colab and Edge Impulse are available. Keep initial data small enough for local storage.

## Engineering rules

- Keep data acquisition, preprocessing, model training, evaluation, and inference as separate, reusable steps. Save all settings with the model so later retraining does not change the output interface.
- Report measured metrics and limits honestly. Do not describe CPU inference or an ONNX file as NPU acceleration. Hailo-10H deployment requires a compatible compiled HEF and runtime validation.
- Never include downloaded audio or environment files in source commits. Keep citations and license information for public datasets.
