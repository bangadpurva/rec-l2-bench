import pandas as pd

from recl2bench.baselines.simple import OracleReranker, PopularityReranker
from recl2bench.rerankers.base import CandidateItem, UserProfile

DAY = 86_400_000
Q = pd.Timestamp("2022-07-16", tz="UTC")
QMS = Q.value // 1_000_000


def test_popularity_counts_only_the_90_days_before_query():
    r = pd.DataFrame({"user_id": list("abcdef"), "parent_asin": ["x", "x", "y", "y", "y", "x"],
                      "rating": 5,
                      "timestamp": [QMS - 1 * DAY, QMS - 10 * DAY,      # x: 2 in window
                                    QMS - 5 * DAY, QMS - 200 * DAY,     # y: 1 in window, 1 too old
                                    QMS + 1 * DAY,                      # y: future, ignored
                                    QMS - 89 * DAY]})                   # x: 3rd in window
    p = UserProfile("u", str(Q), (), (), "t")
    c = [CandidateItem("y", 0, 1.0, ""), CandidateItem("x", 1, 0.9, ""), CandidateItem("z", 2, 0.8, "")]
    assert PopularityReranker(r, 90).rerank(p, c).scores == [1.0, 3.0, 0.0]


def test_oracle():
    p = UserProfile("u", str(Q), (), (), "t")
    c = [CandidateItem("a", 0, 1.0, ""), CandidateItem("b", 1, 0.9, "")]
    assert OracleReranker({"u": {"b"}}).rerank(p, c).scores == [0.0, 1.0]
