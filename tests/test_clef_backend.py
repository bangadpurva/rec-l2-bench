import pytest

from recl2bench.rerankers.decision import (ClefBackend, DecisionReranker, RetryableError,
                                           YesNoQuestion)

Q = YesNoQuestion("yes_no_v1", "Will they buy it?", "matches", "doesn't")


class FakeHTTP:
    def __init__(self, statuses=(200,), envelope=True, field="noul"):
        self.statuses, self.envelope, self.bodies, self.field = list(statuses), envelope, [], field

    def __call__(self, url, headers, body):
        assert url.endswith("/ai/run/@cf/cloudflare/clef-flash")
        assert headers["Authorization"] == "Bearer tok"
        self.bodies.append(body)
        status = self.statuses.pop(0) if self.statuses else 200
        if status != 200:
            return status, {"error": "x"}
        ans = {k: {"type": "noul", self.field: 0.1 + 0.1 * i} for i, k in enumerate(body["questions"])}
        data = {"model": "clef-flash", "answers": ans, "usage": {"input_tokens": 100}}
        return 200, ({"result": data, "success": True} if self.envelope else data)


def test_request_shape_and_parsing():
    http = FakeHTTP(envelope=False)
    be = ClefBackend("acct", "tok", transport=http)
    reply = be.ask("state text", [Q, Q])
    body = http.bodies[0]
    assert body["model"] == "clef-flash" and body["state"] == "state text"
    assert body["questions"]["q0"]["type"] == "noul"
    assert "Answer true if: matches" in body["questions"]["q0"]["instructions"]
    assert reply.probs == pytest.approx([0.1, 0.2]) and reply.model_version == "clef-flash"
    assert be.usage_input_tokens == 100


def test_cloudflare_envelope_unwrapped():
    assert ClefBackend("acct", "tok", transport=FakeHTTP()).ask("s", [Q]).probs == pytest.approx([0.1])


def test_429_is_retryable_and_400_is_not():
    with pytest.raises(RetryableError):
        ClefBackend("acct", "tok", transport=FakeHTTP([429])).ask("s", [Q])
    with pytest.raises(RuntimeError):
        ClefBackend("acct", "tok", transport=FakeHTTP([400])).ask("s", [Q])


def test_question_cap():
    with pytest.raises(ValueError):
        ClefBackend("acct", "tok", transport=FakeHTTP()).ask("s", [Q] * 65)


def test_per_pair_through_adapter_with_retry(profile, candidates):
    be = ClefBackend("acct", "tok", transport=FakeHTTP([429, 529]))
    res = DecisionReranker(be, Q, "per_pair", concurrency=1).rerank(profile, candidates)
    assert res.retries == 2 and res.failures == 0 and res.model_version == "clef-flash"


def test_score_builds_clef_from_its_config(monkeypatch):
    """The path score.py takes: whole YAML config (with its own `name` key) -> backend."""
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    import score
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "tok")
    args = type("A", (), {"scoring_mode": None, "seed": 0})()
    rr, cfg, tsha, mode, conc = score.build("clef", args, {}, root / "configs")
    assert rr.backend.name == "clef-flash" and mode == "per_pair" and tsha
    assert rr.backend.url.endswith("/accounts/acct/ai/run/@cf/cloudflare/clef-flash")


def test_parses_live_response_shape_exactly():
    """Verbatim body from the first live call (2026-10-03)."""
    live = {"result": {"model": "clef-flash", "answers": {"q0": {"type": "noul", "noul": 0.6758}},
                       "usage": {"input_tokens": 166, "output_tokens": 0}},
            "success": True, "errors": [], "messages": []}
    be = ClefBackend("acct", "tok", transport=lambda url, h, body: (200, live))
    reply = be.ask("s", [Q])
    assert reply.probs == pytest.approx([0.6758]) and reply.model_version == "clef-flash"
    assert be.usage_input_tokens == 166


def test_docs_field_name_still_accepted():
    assert ClefBackend("acct", "tok", transport=FakeHTTP(field="probability")).ask("s", [Q]).probs == pytest.approx([0.1])


def test_missing_probability_field_is_an_error():
    bad = {"result": {"model": "clef-flash", "answers": {"q0": {"type": "noul"}}}}
    with pytest.raises(ValueError):
        ClefBackend("acct", "tok", transport=lambda u, h, b: (200, bad)).ask("s", [Q])
