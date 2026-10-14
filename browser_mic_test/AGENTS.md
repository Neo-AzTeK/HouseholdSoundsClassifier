# Browser microphone demo

This folder is a local FastAPI test app for the trained model in `../household_sound_model/`. Keep all app code and notes here. Load the selected checkpoint from the model folder rather than copying it. Browser audio is for live demonstration and room testing; it is not added to the public training set.

The page captures one microphone channel, sends 32-bit float PCM over a WebSocket, and displays the model's five independent scores every roughly 0.5 seconds after the first one-second window. The server applies the same Thingy stream simulation used for public training clips. Browser microphone processing and acoustics still differ from the real Thingy stream, so do not treat these scores as Pi or Hailo validation.

`compose.yaml` runs the app in a CPU-only container. It mounts `audio_model.py`, `predict.py`, and the selected checkpoint read-only from the sibling model folder. The Docker build context is this app folder, so training audio and the model virtual environment are not copied into the image.
