#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
merge_raw_pgn.py

Wrapper around merge_datasets.py for merging PGN-derived CSVs.

Usage:
    python merge_raw_pgn.py --inputs file1.csv file2.csv ... --output dataset_final.csv
"""

import sys
import os

# Add parent directory to path to allow import of merge_datasets
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from merge_datasets import main

if __name__ == '__main__':
    main()
