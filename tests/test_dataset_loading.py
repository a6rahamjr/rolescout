from __future__ import annotations

import pandas as pd
import pytest

from rolescout.data.generator import generate_synthetic_dataset
from rolescout.data.loaders import load_dataset, save_dataset


def test_synthetic_generation_is_reproducible(tmp_path) -> None:
    first = generate_synthetic_dataset(n_queries=20, candidates_per_query=6, seed=7)
    second = generate_synthetic_dataset(n_queries=20, candidates_per_query=6, seed=7)
    pd.testing.assert_frame_equal(first, second)

    path = save_dataset(first, tmp_path / "matches.csv")
    loaded = load_dataset(path)

    assert len(loaded) == 120
    assert loaded["query_id"].nunique() == 20
    assert set(loaded["label"]) == {0, 1}
    assert str(loaded["posted_at"].dtype).startswith("datetime64")


def test_loader_rejects_invalid_schema(tmp_path) -> None:
    path = tmp_path / "invalid.csv"
    pd.DataFrame({"query_id": ["q1"], "label": [1]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_dataset(path)
