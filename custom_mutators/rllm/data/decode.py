"""Decode a u16 token-binary file back to JS source for inspection.

Used to read AFL queue / crashes / hangs files written under the TLAFL
Option B architecture, where AFL files contain little-endian uint16 token
IDs rather than source bytes. Output is the same UTF-8 source AFL hands
to the target via ``post_process``.

Usage:
    python -m data.decode <u16_file>             # JS source to stdout
    python -m data.decode <u16_file> -o <out>    # JS source to <out>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from model.tokenizer import Tokenizer

DEFAULT_TOKENIZER = "Salesforce/codet5p-220m"


def decode_file(path: Path, tokenizer_name: str = DEFAULT_TOKENIZER) -> bytes:
    tokenizer = Tokenizer(tokenizer_name)
    buf = path.read_bytes()
    tokens = tokenizer.parse_u16(buf)
    return tokenizer.detokenize(tokens)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode a u16 token-binary file to JS source.",
    )
    parser.add_argument("input", help="u16 binary file to decode.")
    parser.add_argument(
        "-o", "--output",
        help="Output file (default: stdout).",
    )
    parser.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER,
        help=f"HF tokenizer name or local path (default: {DEFAULT_TOKENIZER}).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    in_path = Path(args.input).expanduser().resolve()
    if not in_path.is_file():
        print(f"[decode] error: not a file: {in_path}", file=sys.stderr)
        sys.exit(2)

    src = decode_file(in_path, tokenizer_name=args.tokenizer)

    if args.output:
        Path(args.output).expanduser().resolve().write_bytes(src)
    else:
        sys.stdout.buffer.write(src)


if __name__ == "__main__":
    main()
