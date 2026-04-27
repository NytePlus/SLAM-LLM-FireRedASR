#!/usr/bin/env python3
"""Read one Kaldi wav entry (ark path with byte offset) and write a WAV file."""

from __future__ import annotations

import argparse
import sys

import kaldiio
import numpy as np
import soundfile as sf


def _to_soundfile_layout(wav: np.ndarray) -> np.ndarray:
    """Kaldi wav matrices are usually (n_channels, n_samples); soundfile wants (n_samples, n_channels)."""
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 1:
        return wav
    if wav.ndim != 2:
        raise ValueError(f"Expected 1D or 2D audio, got shape {wav.shape}")
    if wav.shape[0] == 1:
        return wav[0]
    if wav.shape[1] == 1:
        return wav[:, 0]
    # (channels, samples) -> (samples, channels)
    return wav.T


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Load Kaldi wav ark (e.g. data_wav.ark:12795606) and write tmp.wav by default."
    )
    p.add_argument(
        "ark_spec",
        help="Kaldi specifier: /path/to/file.ark:<byte_offset>",
    )
    p.add_argument(
        "-o",
        "--output",
        default="tmp.wav",
        help="Output WAV path (default: tmp.wav)",
    )
    args = p.parse_args(argv)

    sample_rate, wav_np = kaldiio.load_mat(args.ark_spec)
    wav = _to_soundfile_layout(wav_np)
    sr = int(sample_rate)
    sf.write(args.output, wav, sr, subtype="FLOAT")
    print(f"Wrote {args.output}  sr={sr}  shape={wav.shape}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
