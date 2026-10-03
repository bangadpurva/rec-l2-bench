import numpy as np
import pandas as pd

from recl2bench.l1 import channels as CH

DAY = 86_400_000
Q = pd.Timestamp("2022-07-16", tz="UTC")
QMS = Q.value // 1_000_000
IDX = {a: i for i, a in enumerate("abcde")}


def _cohort(users):
    return pd.DataFrame({"user_id": users, "query_time": Q})


def test_interleave_round_robin_dedup_and_quota():
    ch = {"x": np.array([[0, 1, 2, 3]]), "y": np.array([[1, 4, 5, -1]])}
    idx, src = CH.interleave(ch, 5)
    assert idx[0].tolist() == [0, 1, 2, 4, 3] and src[0].tolist() == ["x", "y", "x", "y", "x"]
    idx, _ = CH.interleave(ch, 4, quotas={"y": 1})
    assert idx[0].tolist() == [0, 1, 2, 3]


def test_cooccurrence_ignores_future_interactions():
    r = pd.DataFrame([
        ("u", "a", 5, QMS - 10 * DAY),          # target user likes a
        ("v", "a", 5, QMS - 20 * DAY), ("v", "b", 5, QMS - 19 * DAY),   # past: a~b
        ("w", "a", 5, QMS - 5 * DAY), ("w", "c", 5, QMS + 5 * DAY),     # future: a~c (must be ignored)
    ], columns=["user_id", "parent_asin", "rating", "timestamp"])
    S = CH.cooccurrence_scores(r, _cohort(["u"]), ["u"], IDX)
    assert S[0, IDX["b"]] > 0
    assert S[0, IDX["c"]] == 0


def test_popularity_window():
    r = pd.DataFrame([("x", "a", 5, QMS - DAY), ("y", "a", 5, QMS - 2 * DAY),
                      ("x", "b", 5, QMS - 200 * DAY), ("x", "c", 5, QMS + DAY)],
                     columns=["user_id", "parent_asin", "rating", "timestamp"])
    S = CH.popularity_scores(r, _cohort(["u"]), ["u"], IDX, 90)
    assert S[0].tolist() == [2, 0, 0, 0, 0]


def test_topm_masks_ineligible():
    S = np.array([[3.0, 2.0, 1.0]])
    idx, _ = CH.topm_from_scores(S, np.array([[False, True, True]]), 3)
    assert idx[0].tolist() == [1, 2, -1]
