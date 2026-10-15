"""Compile ONNX model to Hailo Executable Format (.hef) using Hailo Dataflow Compiler (DFC)."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from household_sounds.hailo.export_onnx import export_onnx
from household_sounds.hailo.calibration import generate_calibration_data

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CHECKPOINT = REPO_ROOT / "artifacts" / "household_sound.pt"
DEFAULT_ONNX = REPO_ROOT / "artifacts" / "household_sound.onnx"
DEFAULT_CALIB = REPO_ROOT / "artifacts" / "calib_set.npy"
DEFAULT_ALLS = REPO_ROOT / "artifacts" / "household_sound.alls"
DEFAULT_HEF = REPO_ROOT / "artifacts" / "household_sound.hef"


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

    print("[*] Running Hailo CLI parser...")
    subprocess.run([
        "hailo", "parser", "onnx", str(onnx_path),
        "--hw-arch", hw_arch,
        "--model-name", "household_sound",
        "--start-node-names", "input",
        "--end-node-names", "logits",
        "--har-path", str(har_path),
    ], check=True)

    print("[*] Running Hailo CLI optimizer...")
    opt_cmd = [
        "hailo", "optimize",
        "--har", str(har_path),
        "--calib-set-path", str(calib_path),
        "--output-har-path", str(quant_har_path),
    ]
    if alls_path and alls_path.is_file():
        opt_cmd.extend(["--model-script", str(alls_path)])
    subprocess.run(opt_cmd, check=True)

    print("[*] Running Hailo CLI compiler...")
    subprocess.run([
        "hailo", "compiler",
        "--har", str(quant_har_path),
        "--hw-arch", hw_arch,
        "--output-dir", str(output_hef_path.parent),
    ], check=True)
    return output_hef_path.is_file()


def print_manual_compilation_instructions(
    onnx_path: Path,
    calib_path: Path,
    alls_path: Path,
    output_hef_path: Path,
    hw_arch: str,
) -> None:
    print("=" * 72)
    print("HAILO DATAFLOW COMPILER (DFC) REQUIRED FOR FINAL HEF COMPILATION")
    print("=" * 72)
    print("The pipeline has verified and prepared all required compilation inputs:")
    print(f"  1. ONNX Model:        {onnx_path}")
    print(f"  2. Calibration Data:  {calib_path} (128 real spectrogram windows)")
    print(f"  3. Model Script:      {alls_path}")
    print(f"  4. Target Hardware:   {hw_arch}")
    print()
    print("To compile the .hef file, run one of the following methods:")
    print()
    print("METHOD A: In an environment with hailo_dataflow_compiler:")
    print(f"  uv run sound-compile --hw-arch {hw_arch}")
    print()
    print("METHOD B: Using Hailo CLI tools:")
    print(f"  hailo parser onnx {onnx_path.name} --hw-arch {hw_arch} --model-name household_sound")
    print(f"  hailo optimize --har household_sound.har --calib-set-path {calib_path.name} --model-script {alls_path.name}")
    print(f"  hailo compiler --har household_sound_quantized.har --hw-arch {hw_arch}")
    print()
    print("METHOD C: Using official Hailo Docker container:")
    print("  docker run --rm -v $(pwd):/workspace -w /workspace \\")
    print("    hailo_dataflow_compiler:latest \\")
    print(f"    python -m household_sounds.hailo.compile --hw-arch {hw_arch}")
    print("=" * 72)


def compile_hef(
    onnx_path: Path = DEFAULT_ONNX,
    calib_path: Path = DEFAULT_CALIB,
    alls_path: Path = DEFAULT_ALLS,
    output_hef_path: Path = DEFAULT_HEF,
    hw_arch: str = "hailo10h",
    strict: bool = False,
) -> Path | None:
    onnx_path = Path(onnx_path)
    calib_path = Path(calib_path)
    alls_path = Path(alls_path)
    output_hef_path = Path(output_hef_path)

    if not onnx_path.is_file():
        print(f"[*] ONNX model not found at {onnx_path}. Exporting...")
        export_onnx(output_path=onnx_path)

    if not calib_path.is_file():
        print(f"[*] Calibration data not found at {calib_path}. Generating...")
        generate_calibration_data(output_path=calib_path)

    has_sdk = False
    try:
        import hailo_sdk_client  # noqa: F401
        has_sdk = True
    except ImportError:
        pass

    if has_sdk:
        return compile_with_hailo_sdk(
            onnx_path=onnx_path,
            calib_path=calib_path,
            alls_path=alls_path,
            output_hef_path=output_hef_path,
            hw_arch=hw_arch,
        )

    cli_success = False
    try:
        cli_success = compile_with_hailo_cli(
            onnx_path=onnx_path,
            calib_path=calib_path,
            alls_path=alls_path,
            output_hef_path=output_hef_path,
            hw_arch=hw_arch,
        )
    except Exception as e:
        print(f"[!] Hailo CLI attempt failed: {e}")

    if cli_success:
        print(f"[+] Successfully compiled with Hailo CLI: {output_hef_path}")
        return output_hef_path

    print_manual_compilation_instructions(
        onnx_path=onnx_path,
        calib_path=calib_path,
        alls_path=alls_path,
        output_hef_path=output_hef_path,
        hw_arch=hw_arch,
    )
    if strict:
        sys.exit(1)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--calib", type=Path, default=DEFAULT_CALIB)
    parser.add_argument("--alls", type=Path, default=DEFAULT_ALLS)
    parser.add_argument("--output", type=Path, default=DEFAULT_HEF)
    parser.add_argument(
        "--hw-arch",
        type=str,
        default="hailo10h",
        choices=["hailo10h", "hailo8", "hailo8l"],
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    compile_hef(
        onnx_path=args.onnx,
        calib_path=args.calib,
        alls_path=args.alls,
        output_hef_path=args.output,
        hw_arch=args.hw_arch,
        strict=args.strict,
    )


if __name__ == "__main__":
    main()
