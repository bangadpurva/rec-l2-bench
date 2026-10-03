"""One manifest.json per run, so every reported number traces to exact inputs."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def sha256_text(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class RunManifest:
    run_id: str
    track: str
    model: str
    model_version: str | None
    scoring_mode: str | None
    pool_sha256: str
    template_sha256: str | None
    budget_tokens: int
    hardware_or_region: str = ""
    concurrency: int | None = None
    truncated_pairs: int = 0
    retries: int = 0
    failures: int = 0
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model_versions_seen: list[str] = field(default_factory=list)

    def write(self, runs_dir: str | Path = "runs") -> Path:
        d = Path(runs_dir) / self.run_id
        d.mkdir(parents=True, exist_ok=True)
        p = d / "manifest.json"
        p.write_text(json.dumps(asdict(self), indent=2) + "\n")
        return p
