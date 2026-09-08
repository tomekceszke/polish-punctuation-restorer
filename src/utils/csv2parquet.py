#!/usr/bin/env python3
"""Polish Punctuation Restorer
Author: Tomasz Ceszke 2026

Converts the CSV splits written by export_dataset.m into Parquet, the format the Hugging Face
dataset viewer reads directly. The CSVs are removed afterwards — they are regenerable.

Requires pyarrow (not needed anywhere else in the project):
    pip install pyarrow

Run from src/:  python3 utils/csv2parquet.py
"""

import os
import sys

import pyarrow.csv as pv
import pyarrow.parquet as pq

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "hf-dataset", "data")


def main():
    for split in ("train", "val", "test"):
        csv_path = os.path.join(DATA_DIR, f"{split}.csv")
        if not os.path.exists(csv_path):
            sys.exit(f"{csv_path} not found — run export_dataset.m first")

        table = pv.read_csv(csv_path)
        parquet_path = os.path.join(DATA_DIR, f"{split}.parquet")
        pq.write_table(table, parquet_path, compression="zstd")
        os.remove(csv_path)

        size = os.path.getsize(parquet_path)
        print(f"{split:<5} {table.num_rows:>8} rows -> {parquet_path} ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
