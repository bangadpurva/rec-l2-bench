import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import diagnose_runs as D  # noqa: E402


def test_diagnose_source_shares_and_positive_ranks():
    # one user, 4 candidates alternating dense/pop in L1 order; positive is the pop item "b"
    pools = pd.DataFrame({"user_id": "u", "rank": range(4), "parent_asin": list("abcd"),
                          "source": ["dense", "pop", "dense", "pop"]})
    pos = pd.DataFrame({"user_id": ["u"], "parent_asin": ["b"]})
    l1 = pd.DataFrame({"user_id": "u", "parent_asin": list("abcd"), "l1_rank": range(4),
                       "score": [4.0, 3.0, 2.0, 1.0]})
    textual = l1.assign(score=[4.0, 1.0, 3.0, 0.0])     # pushes dense items up, b to rank 2
    t = D.diagnose(pools, pos, {"l1_order": l1, "text": textual}, k=2).set_index("model")
    assert t.loc["l1_order", "top2_share_pop"] == pytest.approx(0.5)
    assert t.loc["text", "top2_share_dense"] == pytest.approx(1.0)
    assert t.loc["l1_order", "pos_mean_rank_pop"] == 1
    assert t.loc["text", "pos_mean_rank_pop"] == 2
    assert t.loc["text", "pos_in_top2_pop"] == 0


def test_ties_broken_by_l1_rank():
    s = pd.DataFrame({"user_id": "u", "parent_asin": list("ab"), "l1_rank": [1, 0], "score": [1.0, 1.0]})
    assert D.rank_runs(s).sort_values("new_rank").parent_asin.tolist() == ["b", "a"]
