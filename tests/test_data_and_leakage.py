import pandas as pd
import pytest

from recl2bench.data import leakage_audit as LA
from recl2bench.data.split import TEST_START, first_seen, positives, select_cohort
from recl2bench.l1 import pools as P


def test_cohort_filters_prior_and_positives(ratings):
    c = select_cohort(ratings, "test")
    assert c["user_id"].tolist() == ["u1"]
    assert c.iloc[0].n_prior == 3 and c.iloc[0].n_pos == 2


def test_positives_window(ratings):
    assert positives(ratings, "test") == {"u1": {"d", "e"}, "u2": {"d"}, "u4": {"f"}}


def _pool(items, user="u1"):
    return pd.DataFrame({"user_id": user, "query_time": TEST_START, "rank": range(len(items)),
                         "parent_asin": items, "l1_score": [1.0] * len(items)})


def test_leakage_flags_seen_and_unavailable(ratings):
    fs = first_seen(ratings)
    # u9 has no history; a, b, c were first seen before the test query time
    assert LA.check_pools(_pool(["a", "b", "c"], user="u9"), ratings, fs) == []
    # d is u1's own test positive, but first seen 2022-08-01 -> not yet available
    assert any("not yet available" in e for e in LA.check_pools(_pool(["d"]), ratings, fs))
    errs = LA.check_pools(_pool(["a", "f", "zz"]), ratings, fs)
    assert any("saw before" in e for e in errs)
    assert any("not yet available" in e for e in errs)


def test_forbidden_item_fields():
    assert LA.check_item_fields(pd.DataFrame(columns=["title", "average_rating"]))


def test_pool_freeze_roundtrip_prefix_and_tamper(tmp_path):
    path = tmp_path / "pools_test.parquet"
    digest = P.freeze(_pool(list("defgh")), path)
    df, h = P.load(path, top_m=3)
    assert h == digest and df["parent_asin"].tolist() == ["d", "e", "f"]
    with pytest.raises(FileExistsError):
        P.freeze(_pool(["x"]), path)
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(RuntimeError):
        P.load(path)
