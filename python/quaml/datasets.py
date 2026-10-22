"""PyTorch datasets over event windows. Requires ``torch``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .windows import INPUT_NAMES, GroupIndex, forward_mid_return, model_inputs

# Channels appended to the per-row inputs at gather time.
WINDOW_CHANNELS = INPUT_NAMES + ("log_age", "is_real")


@dataclass
class Sessions:
    """Rows from one or more sessions, concatenated.

    ``group`` is ``session << 32 | instrument_id`` so windows and labels never
    cross a session boundary.
    """

    X: np.ndarray        # [N, F] float32 model inputs
    ts: np.ndarray       # [N] int64 ts_recv
    group: np.ndarray    # [N] int64
    session: np.ndarray  # [N] int32 index into ``paths``
    y: np.ndarray        # [N] float64 forward label, NaN if undefined
    paths: list[Path]

    @property
    def index(self) -> GroupIndex:
        if not hasattr(self, "_index"):
            self._index = GroupIndex(self.group)
        return self._index


def load_sessions(paths, window_ns: int, horizon_ns: int, label_start: str = "anchor") -> Sessions:
    """Read TBBO/TCBBO sessions and build inputs and forward mid-return labels.

    ``label_start`` is passed to :func:`quaml.windows.forward_mid_return`.
    """
    from . import native  # the C++ extension; only market-data loading needs it

    paths = [Path(p) for p in paths]
    parts = []
    for s, path in enumerate(paths):
        ev = native.read_dbn(path)
        feats = native.trade_quote_features(ev, window_ns)
        group = (np.int64(s) << 32) | ev.instrument_id.astype(np.int64)
        y = forward_mid_return(ev.ts_recv, feats["mid"], GroupIndex(group), horizon_ns,
                               start=label_start)
        parts.append((model_inputs(ev, feats), ev.ts_recv, group,
                      np.full(len(ev), s, dtype=np.int32), y))
    X, ts, group, session, y = (np.concatenate(cols) for cols in zip(*parts))
    return Sessions(X, ts, group, session, y, paths)


class EventWindowDataset(Dataset):
    """Samples are ``(x [T, C], mask [T], y)`` for each anchor row.

    ``x`` holds the inputs of the anchor's last ``length`` rows plus
    ``log_age`` (log1p of the time before the anchor, in units of
    ``age_unit`` ts ticks: seconds for nanosecond event data, days for a daily
    panel with ``age_unit=1``) and ``is_real`` (1 for real rows, 0 for
    padding). Padding rows are all-zero.
    """

    def __init__(self, data: Sessions, anchors: np.ndarray, length: int,
                 mean: np.ndarray | None = None, std: np.ndarray | None = None,
                 age_unit: float = 1e9):
        if np.isnan(data.y[anchors]).any():
            raise ValueError("anchors must have defined labels")
        self.data = data
        self.anchors = np.asarray(anchors, dtype=np.int64)
        self.length = length
        self.age_scale = 1.0 / age_unit
        X = data.X if mean is None else (data.X - mean) / std
        self._X = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        self._ts = torch.from_numpy(data.ts)
        self._y = torch.from_numpy(data.y.astype(np.float32))

    def __len__(self) -> int:
        return len(self.anchors)

    def __getitems__(self, items) -> dict[str, torch.Tensor]:
        """Gather a whole batch at once (used by DataLoader)."""
        return self.gather(self.anchors[np.asarray(items)])

    def gather(self, anchors: np.ndarray) -> dict[str, torch.Tensor]:
        """Batch for arbitrary anchor rows of ``data``, not just this dataset's."""
        anchors = np.asarray(anchors, dtype=np.int64)
        idx, mask = self.data.index.windows(anchors, self.length)
        idx = torch.from_numpy(idx)
        mask = torch.from_numpy(mask)
        anchor_t = torch.from_numpy(anchors)

        x = self._X[idx]                                     # [B, T, F]
        age = (self._ts[anchor_t][:, None] - self._ts[idx]).to(torch.float32) * self.age_scale
        x = torch.cat([x, torch.log1p(age)[..., None], mask[..., None].to(torch.float32)], dim=-1)
        x[~mask] = 0.0
        return {"x": x, "mask": mask, "y": self._y[anchor_t], "row": anchor_t}

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        return {k: v[0] for k, v in self.__getitems__([i]).items()}

    def loader(self, batch_size: int, shuffle: bool, **kwargs) -> DataLoader:
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle,
                          collate_fn=lambda batch: batch, **kwargs)


def fit_standardizer(X: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature mean and std over ``rows`` (use training rows only)."""
    mean = X[rows].mean(axis=0)
    std = X[rows].std(axis=0)
    return mean, np.where(std > 0, std, 1.0).astype(np.float32)
