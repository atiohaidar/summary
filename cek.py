"""Simple checker for 'highlights' completeness in CSV files.

Usage:
  python cek.py [path/to/file.csv] [--min-length N] [--limit M]

This script prints IDs whose `highlights` field is considered "incomplete".
Incomplete criteria (configurable):
 - empty string
 - length (after stripping) less than min_length

It also prints a small summary at the end.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd


def find_incomplete_highlights(path: Path, min_length: int = 20, limit: int | None = None) -> list[str]:
	kwargs: dict[str, object] = {"dtype": str}
	if limit is not None and limit > 0:
		kwargs["nrows"] = limit
	try:
		df = pd.read_csv(path, **kwargs)
	except FileNotFoundError:
		raise

	# Ensure columns exist
	if "id" not in df.columns or "highlights" not in df.columns:
		raise ValueError("Input CSV must contain 'id' and 'highlights' columns")

	df = df.fillna("")

	incomplete_ids: list[str] = []
	for idx, row in df.iterrows():
		hid = str(row["id"]) if row["id"] != "" else f"ROW_{idx}"
		highlights = str(row.get("highlights", ""))
		if highlights is None:
			highlights = ""
		text = highlights.strip()
		if text == "" or len(text) < min_length:
			incomplete_ids.append(hid)

	return incomplete_ids


def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description="Check CSV for incomplete 'highlights' and list their IDs")
	parser.add_argument("file", type=Path, nargs="?", default=Path("duc2006.csv"))
	parser.add_argument("--min-length", type=int, default=20, help="Minimum characters for highlights to be considered complete")
	parser.add_argument("--limit", type=int, default=50, help="Number of rows to check (0 = all)")
	args = parser.parse_args(argv)

	limit = None if args.limit == 0 else args.limit

	try:
		incomplete = find_incomplete_highlights(args.file, min_length=args.min_length, limit=limit)
	except FileNotFoundError:
		print(f"File not found: {args.file}", file=sys.stderr)
		return 2
	except ValueError as e:
		print(f"Input error: {e}", file=sys.stderr)
		return 3

	if incomplete:
		print("Found incomplete highlights for the following IDs:")
		for hid in incomplete:
			print(hid)
	else:
		print("No incomplete highlights found.")

	print(f"\nChecked rows: {args.limit if args.limit != 0 else 'all'} | min_length: {args.min_length}")
	print(f"Incomplete count: {len(incomplete)}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())