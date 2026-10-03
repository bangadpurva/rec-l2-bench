"""Item text and user profiles, built only from data before each query time."""
from __future__ import annotations

import pandas as pd

from ..rerankers.base import ProfileItem, UserProfile
from ..rerankers.text import Tokenizer, render_item, render_profile


def _clean(v):
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    if isinstance(v, (list, tuple)):
        v = [x for x in v if x not in (None, "")]
        return v or None
    return v


def item_attrs(meta_row: dict) -> tuple[str, ...]:
    """Two short attributes for profile lines: platform-ish category, then store."""
    cats = _clean(meta_row.get("categories")) or []
    cats = [c for c in cats if c != "Video Games"]
    out = []
    if cats:
        out.append(str(cats[0]))
    if _clean(meta_row.get("store")):
        out.append(str(meta_row["store"]))
    return tuple(out[:2])


def build_items(meta: pd.DataFrame, tok: Tokenizer, item_tokens: int,
                first_seen: pd.Series) -> pd.DataFrame:
    rows = []
    for d in meta.to_dict("records"):
        d = {k: _clean(v) for k, v in d.items()}
        t = render_item(d, tok, item_tokens)
        rows.append({"parent_asin": d["parent_asin"], "title": d.get("title") or "(untitled)",
                     "text": t.text, "n_tokens": t.n_tokens, "truncated": t.truncated,
                     "attrs": list(item_attrs(d))})
    items = pd.DataFrame(rows)
    items["first_seen"] = items["parent_asin"].map(first_seen)
    return items


def build_profiles(ratings: pd.DataFrame, cohort: pd.DataFrame, items: pd.DataFrame,
                   tok: Tokenizer, profile_tokens: int, max_liked: int = 10,
                   max_disliked: int = 3, like_min: float = 4, dislike_max: float = 2):
    """Return (profiles dict, summary DataFrame). Most recent first within each slot."""
    title = items.set_index("parent_asin")["title"].to_dict()
    attrs = items.set_index("parent_asin")["attrs"].to_dict()
    qt_ms = {u: int(q.value // 1_000_000) for u, q in zip(cohort.user_id, cohort.query_time)}
    sub = ratings[ratings.user_id.isin(qt_ms)]
    sub = sub[sub.timestamp < sub.user_id.map(qt_ms)].sort_values("timestamp", ascending=False)

    def pitems(df):
        return tuple(ProfileItem(r.parent_asin, title.get(r.parent_asin, "(unknown)"),
                                 float(r.rating), tuple(attrs.get(r.parent_asin, ())))
                     for r in df.itertuples())

    profiles, summary = {}, []
    for u, g in sub.groupby("user_id", sort=True):
        liked = pitems(g[g.rating >= like_min].head(max_liked))
        disliked = pitems(g[g.rating <= dislike_max].head(max_disliked))
        p = UserProfile(u, str(cohort.set_index("user_id").loc[u, "query_time"]), liked, disliked)
        p, truncated = render_profile(p, tok, profile_tokens)
        profiles[u] = p
        summary.append({"user_id": u, "n_prior": len(g), "n_liked": len(liked),
                        "n_disliked": len(disliked), "profile_truncated": truncated,
                        "text": p.text, "last_event_ms": int(g.timestamp.max())})
    return profiles, pd.DataFrame(summary)
