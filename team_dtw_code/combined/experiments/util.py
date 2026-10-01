"""Shared helpers. Kept out of features.py because that file is a notebook — importing
it would re-execute the whole pipeline (and overwrite the panel) as a side effect."""
from __future__ import annotations

import numpy as np
import pandas as pd


class Px:
    """Exact-timestamp price lookup. Returns NaN when a minute has no bar, which is what
    we want — a missing minute must invalidate a path, never be interpolated."""

    def __init__(self, s: pd.Series):
        self.ts = s.index.values.astype("datetime64[ns]").astype(np.int64)
        self.v = s.to_numpy(float)

    def at(self, stamps) -> np.ndarray:
        q = pd.DatetimeIndex(stamps).tz_convert("UTC").values.astype("datetime64[ns]").astype(np.int64)
        pos = np.searchsorted(self.ts, q)
        pc = np.clip(pos, 0, len(self.ts) - 1)
        out = np.full(len(q), np.nan)
        hit = (pos < len(self.ts)) & (self.ts[pc] == q)
        out[hit] = self.v[pc[hit]]
        return out
