"""Export PyTorch SoundCNN checkpoint to ONNX format for Hailo compilation and inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audio_model import FRAMES, LABELS, PREPROCESSING, SoundCNN


class ExportableSoundModel(nn.Module):
    """Wrapper around SoundCNN that allows optional embedded Sigmoid activation."""

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
    checkpoint_path: Path,
    output_path: Path,
    opset: int = 15,
    include_sigmoid: bool = False,
    batch_size: int = 1,
) -> Path:
    """Export the trained PyTorch checkpoint to ONNX with validation."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[*] Loading PyTorch checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if tuple(checkpoint["labels"]) != LABELS:
        raise ValueError(f"Checkpoint labels {checkpoint['labels']} do not match {LABELS}")

    base_model = SoundCNN().eval()
    base_model.load_state_dict(checkpoint["state_dict"])
    model = ExportableSoundModel(base_model, include_sigmoid=include_sigmoid).eval()

    # Model input shape: [batch_size, 1, 129, FRAMES]
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
        "dynamic_axes": None,  # Static shape is optimal for Hailo hardware compilation
    }
    # PyTorch 2.x supports dynamo=False to use TorchScript-based export
    try:
        torch.onnx.export(model, dummy_input, str(output_path), dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(model, dummy_input, str(output_path), **export_kwargs)

    # 1. Validate ONNX format
    print("[*] Validating ONNX structure...")
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    # 2. Verify with ONNX Runtime against PyTorch
    print("[*] Checking numerical equivalence with ONNX Runtime...")
    ort_session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)

    with torch.no_grad():
        torch_outs = model(dummy_input).numpy()

    max_diff = float(np.max(np.abs(torch_outs - ort_outs[0])))
    print(f"[+] Export successful! Max absolute difference: {max_diff:.3e}")
    if max_diff > 1e-4:
        print(f"[!] Warning: Numerical difference ({max_diff}) is larger than expected.")
    else:
        print("[+] Model output matches PyTorch perfectly.")

    print(f"[+] Output ONNX saved to: {output_path}")
    print(f"    - Input:  {ort_session.get_inputs()[0].name} {ort_session.get_inputs()[0].shape}")
    print(f"    - Output: {ort_session.get_outputs()[0].name} {ort_session.get_outputs()[0].shape}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "artifacts" / "household_sound.pt",
        help="Path to trained .pt file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "household_sound.onnx",
        help="Target ONNX file path",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=15,
        help="ONNX opset version (11, 12, 13, 15, 17)",
    )
    parser.add_argument(
        "--include-sigmoid",
        action="store_true",
        help="Include sigmoid in the ONNX graph instead of raw logits (Hailo prefers raw logits)",
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
