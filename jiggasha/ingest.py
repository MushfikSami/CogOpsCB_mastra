#!/usr/bin/env python3
"""Thin wrapper: redirect to process.py.

Usage:
    python3 ingest.py                    # ingest all rows
    python3 ingest.py --force            # drop & recreate collection
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from process import main

if __name__ == "__main__":
    main()
