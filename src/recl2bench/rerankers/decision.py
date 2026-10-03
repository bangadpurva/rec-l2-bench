"""Decision-model family: Jev, Clef, OpenAI Decisions behind one adapter.

The adapter owns everything model-independent: building the state, the frozen
yes/no question, scoring modes, concurrency, retries on 429/529, and failure
accounting. A Backend owns only the HTTP call and response parsing.

Backends are stubs on purpose: request shapes and the name of the probability
field must be copied from each provider's API reference, not guessed. Implement
`Backend.ask` for each once credentials and docs are in hand.
"""
from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal

from .base import CandidateItem, Reranker, RerankResult, UserProfile

ScoringMode = Literal["per_pair", "fan_out", "listwise"]


@dataclass(frozen=True)
class YesNoQuestion:
    id: str
    question: str
    true_means: str
    false_means: str


@dataclass
class BackendReply:
    probs: list[float]            # P(true) per question, in question order
    model_version: str | None     # the `model` field the response reports


class RetryableError(Exception):
    """Raise from a backend on HTTP 429/529 so the adapter retries."""


class Backend(ABC):
    name: str
    max_questions: int | None = None

    @abstractmethod
    def ask(self, state: str, questions: list[YesNoQuestion]) -> BackendReply: ...


class StubBackend(Backend):
    def __init__(self, name: str, reason: str):
        self.name, self.reason = name, reason

    def ask(self, state, questions):
        raise NotImplementedError(f"{self.name} backend not implemented: {self.reason}")


def make_backend(name: str, **cfg) -> Backend:
    reasons = {
        "jev": "copy request/response shape from TypeSafe API reference (pin jev-1.13.0)",
        "clef": "copy request/response shape from Workers AI @cf/cloudflare/clef docs",
        "openai_decisions": "limited preview; no public spec until access is granted",
    }
    if name not in reasons:
        raise ValueError(f"unknown decision backend {name!r}")
    return StubBackend(name, reasons[name])


def build_state(profile: UserProfile, item: CandidateItem | None = None,
                items: list[CandidateItem] | None = None) -> str:
    s = f"Shopper history\n{profile.text}\n"
    if item is not None:
        s += f"\nCandidate product\n{item.text}\n"
    if items:
        s += "\nCandidate products\n" + "\n\n".join(f"[{c.item_id}]\n{c.text}" for c in items) + "\n"
    return s


class DecisionReranker(Reranker):
    def __init__(self, backend: Backend, question: YesNoQuestion,
                 scoring_mode: ScoringMode = "per_pair", concurrency: int = 8,
                 max_retries: int = 5, shuffle_seed: int | None = None):
        if scoring_mode == "listwise":
            raise NotImplementedError("listwise mode is a Track D variant; add after per_pair works")
        self.backend, self.question = backend, question
        self.scoring_mode, self.concurrency = scoring_mode, concurrency
        self.max_retries, self.shuffle_seed = max_retries, shuffle_seed
        self.name = f"{backend.name}:{scoring_mode}"

    def _call(self, state, questions):
        retries = 0
        while True:
            try:
                return self.backend.ask(state, questions), retries
            except RetryableError:
                retries += 1
                if retries > self.max_retries:
                    raise

    def _score(self, user_profile, candidate_items):
        if self.scoring_mode == "per_pair":
            return self._per_pair(user_profile, candidate_items)
        return self._fan_out(user_profile, candidate_items)

    def _per_pair(self, profile, items):
        def one(c):
            try:
                reply, r = self._call(build_state(profile, item=c), [self.question])
                return reply.probs[0], reply.model_version, r, 0
            except Exception:  # noqa: BLE001 - recorded as a failure, never ranked
                return math.nan, None, 0, 1

        with ThreadPoolExecutor(self.concurrency) as ex:
            out = list(ex.map(one, items))
        versions = {v for _, v, _, _ in out if v}
        return RerankResult(
            scores=[float(s) for s, *_ in out],
            model_version=",".join(sorted(versions)) or None,
            retries=sum(r for *_, r, _ in out), failures=sum(f for *_, f in out),
            extra={"model_versions_seen": sorted(versions)})

    def _fan_out(self, profile, items):
        cap = self.backend.max_questions or len(items)
        order = list(range(len(items)))
        if self.shuffle_seed is not None:      # order-sensitivity check
            random.Random(self.shuffle_seed).shuffle(order)
        scores = [math.nan] * len(items)
        retries = failures = 0
        versions = set()
        for start in range(0, len(order), cap):
            chunk = order[start:start + cap]
            chunk_items = [items[i] for i in chunk]
            qs = [YesNoQuestion(f"{self.question.id}:{c.item_id}",
                                f"For candidate [{c.item_id}]: {self.question.question}",
                                self.question.true_means, self.question.false_means)
                  for c in chunk_items]
            try:
                reply, r = self._call(build_state(profile, items=chunk_items), qs)
                retries += r
                versions.add(reply.model_version)
                for i, p in zip(chunk, reply.probs):
                    scores[i] = float(p)
            except Exception:  # noqa: BLE001
                failures += len(chunk)
        versions.discard(None)
        return RerankResult(scores=scores, model_version=",".join(sorted(versions)) or None,
                            retries=retries, failures=failures,
                            extra={"fan_out_order": order})
