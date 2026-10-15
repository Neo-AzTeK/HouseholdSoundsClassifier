# Household sound model

This project trains a five-score household sound classifier and provides file and live-stream inference. All project files stay in this directory. The current implementation is a **CPU/GPU PyTorch baseline**. A Hailo-10H `.hef` must still be compiled and tested before claiming NPU inference.

## Status

The first trained checkpoint is at `artifacts/household_sound.pt`. It was trained on 1,665 selected public clips with mixed-sound augmentation and passed file and live-adapter smoke tests. Held-out F1 scores range from 0.429 for machines to 0.579 for music; see `RUN_REPORT.md` and `artifacts/household_sound.metrics.json`. These are clip-label proxy metrics, not measured performance in the target room. The original unmixed checkpoint is preserved at `artifacts/household_sound_v1.pt`.

For a step-by-step account of how this checkpoint was trained and how to retrain it, see [TRAINING_EXPLAINED.md](TRAINING_EXPLAINED.md).
For an illustrated Obsidian note with the signal-processing and training maths, see [Household Sound Model.md](docs/Household%20Sound%20Model.md). Open this `household_sound_model` directory as an Obsidian vault so its image embeds and links resolve.

## Output

Each inference update is one JSON object:

```json
{"voices":0.91,"machines":0.08,"collisions":0.03,"steps":0.12,"music":0.76}
```

The five numbers are independent sigmoid scores, so several can be high together. The threshold values used in validation are saved in the checkpoint, but raw scores are emitted for the heatmap when audio is present.
The inference adapter returns five zeros for a window whose decoded RMS is below 0.0005; this quiet gate is saved in the checkpoint.

## Train on the RTX 2060 Super computer

For retraining on the RTX computer, use Python 3.10–3.12. Create the environment, activate it, and install CUDA-enabled PyTorch using [PyTorch's installation selector](https://pytorch.org/get-started/locally/). Then run, from this directory:

```bash
python -m venv .venv
. .venv/bin/activate
# Run the CUDA PyTorch install command from the selector here.
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
.venv/bin/python dataset.py select
.venv/bin/python dataset.py download --max-gb 5
.venv/bin/python dataset.py status
.venv/bin/python train.py --epochs 25
```

On Windows, activate with `.venv\Scripts\activate` and use `.venv\Scripts\python.exe` instead of `.venv/bin/python`. The training run writes `artifacts/household_sound.pt` and `artifacts/household_sound.metrics.json`. Copy those artifacts with this project to the Pi for later CPU inference. The checkpoint is the usable model; do not mistake its `.pt` extension for a Hailo executable.

The selector downloads only a small group of clips from the [FSD50K dataset](https://zenodo.org/records/4060432) through its [per-file mirror](https://huggingface.co/datasets/Fhrozen/FSD50k/tree/main). It records source IDs, original tags, five mapped labels, train/validation/test split, uploader and clip license in `data/manifest.jsonl`. Downloading the full archive is unnecessary. Increase `--train-per-class`, `--val-per-class`, and `--test-per-class` later to expand the subset with the same seed and label mapping. Keep the original FSD50K evaluation split separate for a test set. Each sound file has its own Creative Commons license; consult the manifest before redistribution. If the download stops at the size cap, raise `--max-gb` or select a smaller subset; training requires every manifest clip.

If you copy the audio manually, place each selected clip at its `path` in `data/manifest.jsonl`, for example `data/audio/dev/12345.wav`. You may also copy `dev.csv`, `eval.csv`, `dev_info.json`, and `eval_info.json` into `data/metadata/` before running `dataset.py select`; the script then uses those files without downloading them. `dataset.py status` reports how many selected clips are present. Do not copy random unlabeled files into the audio directory: the manifest is what makes the training labels reproducible.

## Use the model

File input:

```bash
.venv/bin/python predict.py file path/to/sound.wav --summary
```

Without `--summary`, the command prints one score object per overlapping one-second window. With it, the command prints one object containing the maximum score for each category over the file.

Live input:

```bash
your-gateway-chunk-command | .venv/bin/python predict.py live --node-id 0
```

The live command reads one newline-delimited JSON object per decoded Thingy chunk from standard input. Each object has a `node_id` and either `codes` (the 8-bit companded byte values from the BLE packet) or `samples` (decoded float samples). It emits scores every 2,016 samples, about 0.5 seconds. A thin bridge in the existing gateway must forward the chosen node's audio chunks to this command; see `GATEWAY.md`. The file and live commands use the same checkpoint and feature extraction.

## Model and data limits

- One-second windows give about one second of initial audio collection before the first live score. Later scores arrive about every half second. Hardware and gateway delay still need measurement against the one-second goal.
- Public training clips are transformed to approximate the Thingy firmware's 4 kHz compressed stream. The low bandwidth can make impacts and speech/music harder to separate.
- FSD50K labels describe whole clips, not exact sound times. The code chooses loud candidate windows for training, but labels can still be wrong for a particular window. Review the saved test metrics and target-room recordings before using scores as calibrated probabilities.
- The first machine examples are fans, microwaves, printers, and power tools. Dryer and washing-machine coverage requires adding suitable public clips later.
- The current model was trained and timed on this laptop's CPU. It has not been benchmarked on the Pi or compiled for the Hailo-10H.

## Sources

- [FSD50K description, splits, licenses](https://zenodo.org/records/4060432)
- [Per-file FSD50K mirror](https://huggingface.co/datasets/Fhrozen/FSD50k/tree/main)
- [Raspberry Pi AI HAT+ 2 software](https://www.raspberrypi.com/documentation/computers/ai.html)
