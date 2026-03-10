from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
import yaml


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def peak_normalize(waveform: torch.Tensor) -> torch.Tensor:
    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak
    return waveform


def preprocess_file(
    input_path: Path,
    output_path: Path,
    target_sr: int = 16000,
    mono: bool = True,
    normalize: bool = True,
) -> tuple[int, int]:
    waveform, sr = torchaudio.load(str(input_path))

    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=target_sr)

    if normalize:
        waveform = peak_normalize(waveform)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), waveform.squeeze(0).numpy(), target_sr)

    num_samples = waveform.shape[-1]
    return target_sr, num_samples


def find_audio_files(root: Path) -> list[Path]:
    exts = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
    files = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in exts:
            files.append(path)
    return sorted(files)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess audio into mono 16 kHz normalized WAV files.")
    parser.add_argument("--config", default="configs/data.yaml", help="Path to YAML config")
    parser.add_argument("--input-dir", required=True, help="Directory containing raw audio")
    parser.add_argument("--output-dir", required=True, help="Directory to write processed audio")
    args = parser.parse_args()

    cfg = load_config(args.config)
    target_sr = cfg["audio"]["sample_rate"]
    mono = cfg["audio"]["mono"]
    normalize = cfg["audio"]["normalize"] == "peak"

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    audio_files = find_audio_files(input_dir)
    if not audio_files:
        print(f"No audio files found in {input_dir}")
        return

    print(f"Found {len(audio_files)} audio files")

    for src_path in audio_files:
        rel_path = src_path.relative_to(input_dir).with_suffix(".wav")
        dst_path = output_dir / rel_path
        sr, n = preprocess_file(
            input_path=src_path,
            output_path=dst_path,
            target_sr=target_sr,
            mono=mono,
            normalize=normalize,
        )
        print(f"Processed: {src_path} -> {dst_path} | sr={sr} samples={n}")

    print("Done.")


if __name__ == "__main__":
    main()