"""Export PyTorch SoundCNN checkpoint to ONNX format with graph validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from household_sounds.model import FRAMES, LABELS, SoundCNN

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CHECKPOINT = REPO_ROOT / "artifacts" / "household_sound.pt"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "household_sound.onnx"


class ExportableSoundModel(nn.Module):
    """Wrapper around SoundCNN allowing optional embedded Sigmoid activation."""

    def __init__(self, base_model: SoundCNN, include_sigmoid: bool = False) -> None:
        super().__init__()
        self.base_model = base_model
        self.include_sigmoid = include_sigmoid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.base_model(x)
        if self.include_sigmoid:
            return torch.sigmoid(logits)
        return logits


def export_onnx(
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    output_path: Path = DEFAULT_OUTPUT,
    opset: int = 15,
    include_sigmoid: bool = False,
    batch_size: int = 1,
) -> Path:
    """Export the trained PyTorch checkpoint to ONNX with validation."""
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[*] Loading PyTorch checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if tuple(checkpoint["labels"]) != LABELS:
        raise ValueError(f"Checkpoint labels {checkpoint['labels']} do not match {LABELS}")

    base_model = SoundCNN().eval()
    base_model.load_state_dict(checkpoint["state_dict"])
    model = ExportableSoundModel(base_model, include_sigmoid=include_sigmoid).eval()

    dummy_input = torch.randn(batch_size, 1, 129, FRAMES, dtype=torch.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_name = "probabilities" if include_sigmoid else "logits"

    print(f"[*] Exporting to ONNX at {output_path} (opset={opset}, output={output_name})...")
    export_kwargs = {
        "export_params": True,
        "opset_version": opset,
        "do_constant_folding": True,
        "input_names": ["input"],
        "output_names": [output_name],
        "dynamic_axes": None,
    }
    try:
        torch.onnx.export(model, dummy_input, str(output_path), dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(model, dummy_input, str(output_path), **export_kwargs)

    # Validate ONNX structure
    print("[*] Validating ONNX structure...")
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    # Check numerical equivalence with ONNX Runtime
    print("[*] Checking numerical equivalence with ONNX Runtime...")
    ort_session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)

    with torch.no_grad():
        torch_outs = model(dummy_input).numpy()

    max_diff = float(np.max(np.abs(torch_outs - ort_outs[0])))
    print(f"[+] Export successful! Max absolute difference: {max_diff:.3e}")
    print(f"[+] Output ONNX saved to: {output_path}")
    print(f"    - Input:  {ort_session.get_inputs()[0].name} {ort_session.get_inputs()[0].shape}")
    print(f"    - Output: {ort_session.get_outputs()[0].name} {ort_session.get_outputs()[0].shape}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--opset", type=int, default=15)
    parser.add_argument(
        "--include-sigmoid",
        action="store_true",
        help="Include sigmoid activation directly in ONNX graph (Hailo prefers raw logits)",
    )
    args = parser.parse_args()

    export_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        opset=args.opset,
        include_sigmoid=args.include_sigmoid,
    )


if __name__ == "__main__":
    main()
