#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
merge_datasets.py

Concatenates multiple processed CSV files into a single dataset.

Usage:
    python merge_datasets.py --inputs file1.csv file2.csv ... --output dataset_final.csv
"""

import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple processed CSV files into one dataset."
    )
    parser.add_argument('--inputs', nargs='+', required=True,
                        help='List of input CSV files to merge')
    parser.add_argument('--output', required=True,
                        help='Path to the output CSV file')
    args = parser.parse_args()

    frames = []
    for fpath in args.inputs:
        df = pd.read_csv(fpath)
        frames.append(df)

    if frames:
        merged = pd.concat(frames, ignore_index=True)
        merged.to_csv(args.output, index=False)
        print(f"Merged {len(frames)} files into {args.output}")
    else:
        print("No input files provided.")


if __name__ == '__main__':
    main()
