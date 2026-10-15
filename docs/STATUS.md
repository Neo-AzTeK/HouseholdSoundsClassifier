# Execution status

- [x] Project directory and `AGENTS.md` created.
- [x] Reproducible FSD50K subset selector and per-clip downloader written.
- [x] Stream-matched audio preprocessing, five-output CNN, training, evaluation, and inference commands written.
- [x] Dataset mapping smoke-tested and firmware FIR/companding tables checked against `thingy-ranging`.
- [x] Public audio downloaded into `data/audio/`: 1,665 FSD50K clips, 1.2 GB; all clips opened as mono WAVs.
- [x] Dependencies installed in the Python 3.12 project environment (CPU PyTorch).
- [x] Real checkpoint trained and evaluated; selected mixed-audio model and original baseline are documented in `RUN_REPORT.md`.
- [x] File inference run with trained checkpoint; live NDJSON adapter also smoke-tested.
- [x] Digital silence and very faint noise return five zeros through the saved quiet gate.
- [ ] Live Thingy gateway bridge connected and latency measured.
- [ ] Hailo-10H model compiled to `.hef` and validated on the Pi.

The user downloaded the public clips and installed dependencies from a network-enabled terminal. The first checkpoint is `artifacts/household_sound.pt`. Actual Thingy gateway integration, target-room validation, Pi timing, and Hailo HEF compilation remain.
