"""Ingestion: load the config and raw data, and fingerprint the data file."""
import hashlib                # built-in: makes hashes (fingerprints)
from pathlib import Path      # built-in: works with file paths

import pandas as pd
import yaml


def load_config(path="config/config.yaml"):
    """Read the YAML config file and return it as a Python dictionary."""
    with open(path) as f:          # open the file; 'with' closes it automatically
        return yaml.safe_load(f)   # turn YAML text into a dictionary


def load_raw(path):
    """Load the raw CSV, with a clear error if it's missing."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Raw data not found at '{path}'. Check raw_path in config/config.yaml."
        )
    return pd.read_csv(path)


def file_sha256(path):
    """Return the SHA-256 fingerprint of a file. Same file = same hash."""
    h = hashlib.sha256()                   # an empty hasher
    with open(path, "rb") as f:            # "rb" = read in binary (raw bytes)
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)                # feed the file in 1 MB pieces
    return h.hexdigest()                   # the fingerprint as a text string
if __name__ == "__main__":
    cfg = load_config()
    path = cfg["data"]["raw_path"]
    df = load_raw(path)
    print(f"Loaded {len(df)} rows")
    print(f"Dataset version: {file_sha256(path)[:12]}")