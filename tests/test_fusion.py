import pandas as pd
import pytest

from recl2bench import fusion as F


def _run(scores):
    return pd.DataFrame({"user_id": "u", "parent_asin": list("abcd"), "l1_rank": range(4), "score": scores})


def test_endpoints_reproduce_inputs():
    rr, pt = _run([1.0, 4.0, 3.0, 2.0]), _run([4.0, 3.0, 2.0, 1.0])
    order = lambda df: df.sort_values(["score", "l1_rank"], ascending=[False, True]).parent_asin.tolist()  # noqa: E731
    assert order(F.fuse_scores(rr, pt, 1.0)) == ["b", "c", "d", "a"]
    assert order(F.fuse_scores(rr, pt, 0.0)) == ["a", "b", "c", "d"]


def test_fusion_can_beat_both_inputs():
    # positive "c" is rank 3 for both inputs, but the inputs disagree on everything above it
    rr, pt = _run([4.0, 1.0, 3.0, 2.0]), _run([1.0, 4.0, 3.0, 2.0])
    pos = {"u": {"c"}}
    w, grid = F.tune(rr, pt, pos)
    assert grid.set_index("w").loc[0.5, "ndcg@10"] > grid.set_index("w").loc[0.0, "ndcg@10"]
    assert grid.set_index("w").loc[0.5, "ndcg@10"] > grid.set_index("w").loc[1.0, "ndcg@10"]
    assert 0 < w < 1


def test_ties_choose_smaller_weight():
    rr = pt = _run([4.0, 3.0, 2.0, 1.0])
    w, _ = F.tune(rr, pt, {"u": {"a"}})
    assert w == 0.0


def test_mismatched_candidates_rejected():
    with pytest.raises(ValueError):
        F.fuse_scores(_run([1.0, 2, 3, 4]), _run([1.0, 2, 3, 4]).iloc[:3], 0.5)
