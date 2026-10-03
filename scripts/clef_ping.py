"""One raw Clef call: prints the HTTP status and response body, so auth or format
problems are visible. Reads CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN from .env.

    python scripts/clef_ping.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import yaml

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
from recl2bench.rerankers.decision import ClefBackend, YesNoQuestion  # noqa: E402

env = root / ".env"
if env.exists():
    for line in env.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

cfg = yaml.safe_load((root / "configs/models/clef.yaml").read_text())
acct, tok = os.environ.get("CLOUDFLARE_ACCOUNT_ID", ""), os.environ.get("CLOUDFLARE_API_TOKEN", "")
print(f"account id: {'set, ' + str(len(acct)) + ' chars' if acct else 'MISSING'}")
print(f"api token:  {'set, ' + str(len(tok)) + ' chars' if tok else 'MISSING'}")
if not (acct and tok):
    sys.exit("fill both in .env (no quotes, no spaces)")

be = ClefBackend(acct, tok, cfg["model_version"])
q = YesNoQuestion("ping", "Would this shopper buy guitar strings?", "yes", "no")
body = {"model": be.name, "state": "Shopper bought an acoustic guitar and a capo.",
        "questions": {"q0": {"type": "noul", "instructions": be.instructions(q)}}}
status, data = be._post(be.url, {"Authorization": f"Bearer {tok}", "Content-Type": "application/json"}, body)
print(f"POST {be.url.replace(acct, '<account>')}\nHTTP {status}")
print(json.dumps(data, indent=2)[:3000])
