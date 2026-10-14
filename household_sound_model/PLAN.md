# Household sound classifier plan

Last updated: 2026-10-12

The first CPU checkpoint has now been trained. See `RUN_REPORT.md` for the measured results and `STATUS.md` for remaining deployment work.

## Goal

Train a first demonstrable model that takes one audio channel and returns five scores. The eventual runtime is a Raspberry Pi 5 with an AI HAT+ 2 (Hailo-10H), but the immediate focus is **model training and an audio-to-scores demo**, not integration with the four-microphone gateway. The five target categories are:

1. **Voices** — audible speech and voices, including voices played through a speaker.
2. **Machines** — sustained appliance and machine sounds, such as a dryer or washing machine.
3. **Collisions** — objects hitting other objects or surfaces.
4. **Steps** — people walking.
5. **Music** — audible music from a speaker or other source. Singing over music may score high for both voices and music.

The scores will eventually drive the color of an existing heatmap. The Raspberry Pi also receives four microphones and performs other sound-localization computations; classification only needs one audio channel. Choosing a particular microphone is outside the current model-training scope. The device is in one room at a fixed position.

## Decisions and answers so far

| Topic | Decision |
| --- | --- |
| Compute hardware | Raspberry Pi 5 with AI HAT+ 2 (Hailo-10H). |
| Microphone | One Nordic Thingy:53 microphone channel from the four-microphone receiver. Any one channel is sufficient for model development. |
| Training data | Start with public datasets only. |
| Local data | Short recordings from the target room may be used for testing, but not initially for training. |
| Latency | Aim for a result within about one second of a sound. |
| Output cadence | Scores every 0.5–1 second. |
| Downstream use | Heatmap colors; no notification or event log is required. |
| Simultaneous sounds | Use five independent scores so more than one category can be present. Speech or singing played through a speaker can count as both voices and music when both are audible. |
| Audio transport/format | Existing firmware streams each Thingy:53 microphone through BLE notifications to a Python gateway. The stream is about 4,031 samples/s per node after filtering, 4:1 decimation, and 8-bit compression. Pi operation still needs testing. |
| Firmware scope | Keep the current audio stream unchanged for the first model. The user wants a demonstrable training result first. |
| Training resources | Use local Python on a second computer with an NVIDIA RTX 2060 Super for the main training run. The current computer has an AMD Radeon RX 6800S. Edge Impulse and substantial Google Colab access are also available. |
| Local storage | About 30–32 GB free. Keep the first dataset and derived features well below that limit; do not download and extract all of FSD50K locally. |
| Dataset size strategy | Start with a selected public-data subset and a reproducible selection manifest; expanding the subset later should require adding source files or manifest rows, not changing labels or inference code. |
| First demo | Support both audio files and a live Thingy:53 stream with the same model and preprocessing. |
| Collision rule | Include all short impacts, including objects hitting each other, dropped objects, and doors slamming. |
| First demo priority | Show all five categories working. Accuracy and latency will still be measured, but a proof of all five takes priority for the first iteration. |
| Data handoff | The user can copy public audio into this project directory. The subset selector downloads FSD50K metadata and writes the exact clip manifest; the audio must then be downloaded or copied before training. |

Expected output shape (one object per update):

```json
{
  "voices": 0.91,
  "machines": 0.08,
  "collisions": 0.03,
  "steps": 0.12,
  "music": 0.76
}
```

Scores should lie between 0 and 1. We will decide later whether the heatmap needs calibrated probabilities or only stable relative scores. The output object does not need timestamps unless the consuming interface later requires them.

## Existing audio path in `thingy-ranging`

- Each Thingy:53 captures approximately 16.125 kHz mono audio in 160-sample buffers. The continuous streaming firmware filters and decimates by four, then compresses to 8-bit samples. The resulting per-node stream is approximately 4,031 samples/s, with about 2 kHz of usable bandwidth. See [sample configuration](/home/mixcraftio/Code/Repos/thingy-ranging/thingy_ranging/include/ranging.h:62) and [stream processing](/home/mixcraftio/Code/Repos/thingy-ranging/thingy_ranging/src/stream.c:1).
- Chunks travel in BLE notifications through the central board. The Python gateway decodes them into per-node sample arrays, so the classifier can select one node. See [chunk format](/home/mixcraftio/Code/Repos/thingy-ranging/thingy_ranging/include/conn_time_sync.h:161) and [decoder](/home/mixcraftio/Code/Repos/thingy-ranging/thingy_ranging/tools/noise.py:186).
- The classifier needs its own gateway consumer and must keep streaming enabled even when the existing listening or localization views are inactive. See [gateway dispatch](/home/mixcraftio/Code/Repos/thingy-ranging/thingy_ranging/tools/gateway.py:231) and [stream control](/home/mixcraftio/Code/Repos/thingy-ranging/thingy_ranging/tools/gateway.py:912).
- The firmware produces a chunk about every 47 ms. Two-board streaming has been tested on a laptop; four-board throughput and Pi operation remain to be measured. See [stream timing](/home/mixcraftio/Code/Repos/thingy-ranging/thingy_ranging/src/stream.c:39) and [project status](/home/mixcraftio/Code/Repos/thingy-ranging/thingy_ranging/STATUS.md:28).
- **Model implication:** upsampling this stream to 16 kHz does not restore the missing high frequencies. For the first model, simulate the deployed bandwidth on public training and validation audio. This is a quality constraint to measure, not a prerequisite for starting training. The existing stream will not be changed for this first version.

## Draft implementation path

1. **Fix label rules.** Use audible content: TV speech counts as voices, and singing over music can have both labels. Collisions include short impacts such as dropped objects and slamming doors. Decide the remaining boundaries for footsteps, appliance noises, silence, and other sounds.
2. **Prepare a storage-limited public dataset.** Start from FSD50K's existing speech, music, footsteps, impact, and machine-related labels, plus a small appliance supplement if needed. Candidate FSD50K tags include `Speech`, `Human voice`, `Music`, `Walk, footsteps`, `Slam`, `Knock`, `Thump, thud`, `Mechanical fan`, and `Microwave oven` ([taxonomy](https://fsannotator.upf.edu/fsd/release_table/FSD50K/taxonomy_table/)). Audit the precise tag mapping before assigning labels. Select an initial target of roughly 200–400 distinct positive clips per category plus negative/background clips, subject to availability and label quality. Keep a manifest with source, original clip ID, source tags, five mapped labels, split, and license. When expanding, append clips through this manifest using the same preprocessing and split rules. Avoid downloading and extracting the full FSD50K archive on the current computer; its official development audio is about 18 GB compressed, before extraction ([download page](https://zenodo.org/records/4060432)).
3. **Prepare audio windows.** Transform public clips to approximate the deployed filtering, decimation, and compression. Avoid mixing samples from the same original recording or uploader between training and validation. FSD50K's labels apply to whole clips, so do not assume every short window contains every clip label; inspect or filter candidate windows, particularly for brief impacts and steps. Save a compact processed cache and the exact preprocessing settings.
4. **Train a baseline.** Use local Python on the RTX 2060 Super computer. Compare a pretrained audio model such as YAMNet with a compact log-mel audio classifier trained or fine-tuned on stream-matched public clips. YAMNet expects 16 kHz input, so its use here requires resampling and must be validated against the reduced bandwidth. Use short overlapping windows and five independent outputs. Export the best model. Edge Impulse is available for experiments, but the main workflow should preserve direct control of multi-label targets, stream-matched preprocessing, and later Hailo-10H conversion.
5. **Evaluate.** Measure precision and recall per category and false positives on background audio using held-out public data. Then use permitted target-room recordings as a separate, realistic test set. Check where the 2 kHz bandwidth hurts results.
6. **Demonstrate.** Provide one inference function that accepts an audio window and returns the five-score JSON object. Give it two input adapters: a file reader and a live Thingy stream reader. The live adapter adds buffering and a recurring 0.5–1 second update; it does not change how the model is trained. Keep audio preprocessing, label mapping, thresholds, and model weights as versioned components so a later model can be retrained and swapped without changing the output interface.
7. **Later deployment.** After the model demo works, measure inference on the Pi. If NPU acceleration is worthwhile, compile a Hailo-10H-compatible model to HEF and compare accuracy and latency with the original. Connecting it to the gateway and heatmap is a subsequent phase.

## Open points

- Final public-data acquisition route for just the selected clips, and whether an appliance supplement is needed after reviewing FSD50K labels.
- How much performance is needed to call the first demo successful.
- Whether a pretrained audio model can be compiled directly for Hailo-10H; the CPU baseline is the first implementation checkpoint.
- Live four-node throughput, gateway hookup, and heatmap integration are deferred until after the model demo.

## References

- [Raspberry Pi AI HAT hardware](https://www.raspberrypi.com/documentation/accessories/ai-hat-plus.html)
- [Raspberry Pi Hailo software setup](https://www.raspberrypi.com/documentation/computers/ai.html)
- [Nordic Thingy:53 product overview](https://www.nordicsemi.com/Products/Development-hardware/Nordic-Thingy-53?lang=en)
- [FSD50K public dataset](https://fsannotator.upf.edu/fsd/release/FSD50K/)
- [FSD50K download sizes and licensing](https://zenodo.org/records/4060432)
- [TensorFlow YAMNet transfer-learning tutorial](https://www.tensorflow.org/tutorials/audio/transfer_learning_audio)
- [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo)
- [AMD ROCm compatibility](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)
