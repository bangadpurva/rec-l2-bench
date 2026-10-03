"""Cross-encoder family: bge-reranker-v2-m3 and Qwen3-Reranker-0.6B.

Not yet run end to end (needs a GPU and model downloads). The BGE path follows
its model card (sigmoid of the single logit). The Qwen path follows the
Qwen3-Reranker model card's yes/no prompt; verify it against the pinned
revision's card before the first real run.
"""
from __future__ import annotations

from .base import CandidateItem, Reranker, RerankResult, UserProfile


def pick_device(torch, requested: str | None = None) -> str:
    """CUDA, then Apple GPU (MPS), then CPU."""
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(torch, device: str):
    return torch.float16 if device in ("cuda", "mps") else torch.float32


class _HFBase(Reranker):
    def __init__(self, hf_model: str, revision: str | None, max_length: int = 512,
                 batch_size: int = 32, device: str | None = None):
        import torch  # lazy: keeps the core package importable without torch
        self.torch = torch
        self.hf_model, self.revision = hf_model, revision
        self.max_length, self.batch_size = max_length, batch_size
        self.device = pick_device(torch, device)
        self.dtype = pick_dtype(torch, self.device)
        self.truncated_pairs = 0

    def _batches(self, items):
        for i in range(0, len(items), self.batch_size):
            yield items[i:i + self.batch_size]


class BGEReranker(_HFBase):
    name = "bge_reranker"

    def __init__(self, hf_model="BAAI/bge-reranker-v2-m3", revision=None, **kw):
        super().__init__(hf_model, revision, **kw)
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(hf_model, revision=revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            hf_model, revision=revision, dtype=self.dtype).to(self.device).eval()

    def _score(self, user_profile: UserProfile, candidate_items: list[CandidateItem]):
        q = user_profile.text
        scores = []
        with self.torch.no_grad():
            for batch in self._batches(candidate_items):
                pairs = [[q, c.text] for c in batch]
                full = self.tok(pairs, truncation=False)["input_ids"]
                self.truncated_pairs += sum(len(x) > self.max_length for x in full)
                enc = self.tok(pairs, padding=True, truncation=True,
                               max_length=self.max_length, return_tensors="pt").to(self.device)
                logits = self.model(**enc).logits.view(-1).float()
                scores += self.torch.sigmoid(logits).cpu().tolist()
        return RerankResult(scores=[float(s) for s in scores],
                            model_version=f"{self.hf_model}@{self.revision}",
                            extra={"device": self.device})


class Qwen3Reranker(_HFBase):
    name = "qwen3_reranker"
    PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on "
              "the Query and the Instruct provided. Note that the answer can only be \"yes\" or "
              "\"no\".<|im_end|>\n<|im_start|>user\n")
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __init__(self, instruction: str, hf_model="Qwen/Qwen3-Reranker-0.6B", revision=None, **kw):
        super().__init__(hf_model, revision, **kw)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.instruction = instruction
        self.tok = AutoTokenizer.from_pretrained(hf_model, revision=revision, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_model, revision=revision, dtype=self.dtype).to(self.device).eval()
        self.yes_id = self.tok.convert_tokens_to_ids("yes")
        self.no_id = self.tok.convert_tokens_to_ids("no")

    def _score(self, user_profile, candidate_items):
        scores = []
        with self.torch.no_grad():
            for batch in self._batches(candidate_items):
                texts = [f"{self.PREFIX}<Instruct>: {self.instruction}\n<Query>: {user_profile.text}"
                         f"\n<Document>: {c.text}{self.SUFFIX}" for c in batch]
                enc = self.tok(texts, padding=True, return_tensors="pt").to(self.device)
                last = self.model(**enc).logits[:, -1, :]
                two = self.torch.stack([last[:, self.no_id], last[:, self.yes_id]], dim=1)
                scores += self.torch.softmax(two.float(), dim=1)[:, 1].cpu().tolist()
        return RerankResult(scores=[float(s) for s in scores],
                            model_version=f"{self.hf_model}@{self.revision}")
