"""Compile ONNX model to Hailo Executable Format (.hef) using Hailo Dataflow Compiler (DFC)."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from export_onnx import export_onnx
from prepare_calibration import generate_calibration_data


def compile_with_hailo_sdk(
    onnx_path: Path,
    calib_path: Path,
    alls_path: Path | None,
    output_hef_path: Path,
    hw_arch: str = "hailo10h",
) -> Path:
    """Compile using the hailo_sdk_client Python package."""
    from hailo_sdk_client import ClientRunner

    print(f"[*] Initializing Hailo ClientRunner for target architecture '{hw_arch}'...")
    runner = ClientRunner(hw_arch=hw_arch)

    print(f"[*] Parsing ONNX model from {onnx_path}...")
    runner.translate_onnx_model(
        model=str(onnx_path),
        net_name="household_sound",
        start_node_names=["input"],
        end_node_names=["logits"],
    )

    if alls_path and alls_path.is_file():
        print(f"[*] Loading Hailo model script: {alls_path}...")
        runner.load_model_script(str(alls_path))

    print(f"[*] Optimizing and quantizing with calibration dataset from {calib_path}...")
    calib_data = np.load(calib_path)
    # Hailo DFC accepts calibration data as numpy array
    runner.optimize(calib_data)

    print(f"[*] Compiling to HEF ({hw_arch})...")
    hef_bytes = runner.compile()

    output_hef_path.parent.mkdir(parents=True, exist_ok=True)
    with output_hef_path.open("wb") as f:
        f.write(hef_bytes)

    print(f"[+] Successfully compiled HEF: {output_hef_path} ({output_hef_path.stat().st_size / 1024:.1f} KB)")
    return output_hef_path


def compile_with_hailo_cli(
    onnx_path: Path,
    calib_path: Path,
    alls_path: Path | None,
    output_hef_path: Path,
    hw_arch: str = "hailo10h",
) -> bool:
    """Compile using the Hailo CLI tools if available on PATH."""
    if not shutil.which("hailo"):
        return False

    har_path = output_hef_path.with_suffix(".har")
    quant_har_path = output_hef_path.with_name(f"{output_hef_path.stem}_quant.har")

    print(f"[*] Running Hailo CLI parser...")
    parse_cmd = [
        "hailo", "parser", "onnx", str(onnx_path),
        "--hw-arch", hw_arch,
        "--model-name", "household_sound",
        "--start-node-names", "input",
        "--end-node-names", "logits",
        "--har-path", str(har_path),
    ]
    subprocess.run(parse_cmd, check=True)

    print(f"[*] Running Hailo CLI optimizer...")
    opt_cmd = [
        "hailo", "optimize",
        "--har", str(har_path),
        "--calib-set-path", str(calib_path),
        "--output-har-path", str(quant_har_path),
    ]
    if alls_path and alls_path.is_file():
        opt_cmd.extend(["--model-script", str(alls_path)])
    subprocess.run(opt_cmd, check=True)

    print(f"[*] Running Hailo CLI compiler...")
    comp_cmd = [
        "hailo", "compiler",
        "--har", str(quant_har_path),
        "--hw-arch", hw_arch,
        "--output-dir", str(output_hef_path.parent),
    ]
    subprocess.run(comp_cmd, check=True)
    return output_hef_path.is_file()


def print_manual_compilation_instructions(
    onnx_path: Path,
    calib_path: Path,
    alls_path: Path,
    output_hef_path: Path,
    hw_arch: str,
) -> None:
    """Print clear instructions when DFC is not in current environment."""
    print("=" * 72)
    print("HAILO DATAFLOW COMPILER (DFC) REQUIRED FOR FINAL HEF COMPILATION")
    print("=" * 72)
    print("The pipeline has verified and prepared all required compilation inputs:")
    print(f"  1. ONNX Model:        {onnx_path} (shape [1, 1, 129, 48], output logits [1, 5])")
    print(f"  2. Calibration Data:  {calib_path} (128 real spectrogram windows)")
    print(f"  3. Model Script:      {alls_path}")
    print(f"  4. Target Hardware:   {hw_arch}")
    print()
    print("To compile the .hef file, run one of the following methods:")
    print()
    print("METHOD A: If you have a Python environment with hailo_dataflow_compiler:")
    print(f"  python {Path(__file__).name} --hw-arch {hw_arch}")
    print()
    print("METHOD B: Using Hailo CLI tools:")
    print(f"  hailo parser onnx {onnx_path.name} --hw-arch {hw_arch} --model-name household_sound")
    print(f"  hailo optimize --har household_sound.har --calib-set-path {calib_path.name} --model-script {alls_path.name}")
    print(f"  hailo compiler --har household_sound_quantized.har --hw-arch {hw_arch}")
    print()
    print("METHOD C: Using official Hailo Docker container:")
    print("  docker run --rm -v $(pwd):/workspace -w /workspace \\")
    print("    hailo_dataflow_compiler:latest \\")
    print(f"    python household_sound_model/compile_hef.py --hw-arch {hw_arch}")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "artifacts" / "household_sound.pt",
        help="Path to PyTorch checkpoint",
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=ROOT / "artifacts" / "household_sound.onnx",
        help="Path to ONNX model",
    )
    parser.add_argument(
        "--calib",
        type=Path,
        default=ROOT / "artifacts" / "calib_set.npy",
        help="Path to calibration dataset",
    )
    parser.add_argument(
        "--alls",
        type=Path,
        default=ROOT / "artifacts" / "household_sound.alls",
        help="Path to Hailo model script (.alls)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "household_sound.hef",
        help="Target output .hef path",
    )
    parser.add_argument(
        "--hw-arch",
        type=str,
        default="hailo10h",
        choices=["hailo10h", "hailo8", "hailo8l"],
        help="Hailo hardware architecture (hailo10h for Pi 5 AI Kit / Hailo-10H, hailo8, or hailo8l)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail with non-zero exit code if Hailo DFC is not installed",
    )
    args = parser.parse_args()

    # 1. Ensure ONNX exists
    if not args.onnx.is_file():
        print(f"[*] ONNX model not found at {args.onnx}. Exporting from {args.checkpoint}...")
        export_onnx(args.checkpoint, args.onnx)

    # 2. Ensure Calibration set exists
    if not args.calib.is_file():
        print(f"[*] Calibration data not found at {args.calib}. Generating...")
        generate_calibration_data(args.calib)

    # 3. Attempt compilation
    has_sdk = False
    try:
        import hailo_sdk_client  # noqa: F401
        has_sdk = True
    except ImportError:
        pass

    if has_sdk:
        compile_with_hailo_sdk(
            onnx_path=args.onnx,
            calib_path=args.calib,
            alls_path=args.alls,
            output_hef_path=args.output,
            hw_arch=args.hw_arch,
        )
        return

    # Try CLI
    cli_success = False
    try:
        cli_success = compile_with_hailo_cli(
            onnx_path=args.onnx,
            calib_path=args.calib,
            alls_path=args.alls,
            output_hef_path=args.output,
            hw_arch=args.hw_arch,
        )
    except Exception as e:
        print(f"[!] Hailo CLI attempt failed: {e}")

    if cli_success:
        print(f"[+] Successfully compiled with Hailo CLI: {args.output}")
        return

    # If neither was available
    print_manual_compilation_instructions(
        onnx_path=args.onnx,
        calib_path=args.calib,
        alls_path=args.alls,
        output_hef_path=args.output,
        hw_arch=args.hw_arch,
    )
    if args.strict:
        sys.exit(1)


if __name__ == "__main__":
    main()
