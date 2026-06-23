from __future__ import annotations

from typing import Any, Iterable

import pandas as pd


def iter_csv_rows(path: str) -> Iterable[dict[str, Any]]:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    cols = list(df.columns)
    for _, row in df.iterrows():
        d = {c: row.get(c, "") for c in cols}
        yield d

