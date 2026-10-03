# Running the benchmark on Vast.ai

Everything runs from one script, `scripts/run_pipeline.sh`, one stage at a time.
Each stage saves its output to `logs/`. Stages marked **STOP** produce numbers that
decide the next step: send the output for review before moving on.

## 1. Rent the instance

| Setting | Choose |
|---|---|
| GPU | 1× 48 GB card: L40S, RTX 6000 Ada or A6000 |
| System RAM | 32 GB or more (co-occurrence on larger categories is RAM-bound) |
| Disk | 100 GB |
| Rental type | **On-demand**, not interruptible |
| Host | reliability > 99%, verified / secure cloud |
| Template | PyTorch (CUDA), any recent version |

Only one GPU is needed; nothing here uses more than one.

## 2. Set up (once, ~5 minutes)

Open the instance's terminal (or SSH in), then:

```bash
cd /workspace
git clone https://github.com/bangadpurva/rec-l2-bench.git
cd rec-l2-bench
bash scripts/vast_setup.sh
```

`git clone` asks for your GitHub username and a personal access token (the repo is
private). Setup installs everything, prints the GPU, and runs the test suite: it should
end with all tests passed.

Then put API keys in `.env` (created from `.env.example`). At minimum set `HF_TOKEN`;
Clef needs `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN`.

```bash
nano .env
```

Always work inside tmux, so a dropped connection does not kill a run:

```bash
tmux new -s bench        # reattach later with: tmux attach -t bench
```

## 3. Stages

Set the category once per terminal (after the screen picks it):

```bash
export CAT=Musical_Instruments
```

| # | Command | What it does | GPU | Time (rough) |
|---|---|---|---|---|
| 1 | `bash scripts/run_pipeline.sh screen` | Scores 4 candidate categories on validation. **STOP** | no | 10–15 min |
| 2 | `bash scripts/run_pipeline.sh data` | Downloads `$CAT`, builds 2,000 valid / 3,000 test users | no | 5 min |
| 3 | `bash scripts/run_pipeline.sh l1-tune` | Embeds the catalog, tunes the half-life. **STOP** | yes | 5–15 min |
| 4 | `HL=<days> bash scripts/run_pipeline.sh l1-compare` | Recall per L1 channel and merged pools. **STOP** | no | 5–10 min |
| 5 | `HL=<days> CHANNELS=<mix> bash scripts/run_pipeline.sh l1-dryrun` | Builds pools and the report, freezes nothing. **STOP** | no | 5 min |
| 6 | `HL=<days> CHANNELS=<mix> bash scripts/run_pipeline.sh l1-freeze` | Freezes pools and eval cohort. Asks you to type FREEZE. Irreversible | no | 5 min |
| 7 | `bash scripts/run_pipeline.sh backup` | Archives processed data, pools, runs, results | no | 1 min |
| 8 | `bash scripts/run_pipeline.sh baselines` | L1 order, random, popularity, oracle | no | 2 min |
| 9 | `bash scripts/run_pipeline.sh bge` | bge-reranker-v2-m3 | yes | 10–20 min |
| 10 | `bash scripts/run_pipeline.sh qwen3` | Qwen3-Reranker-0.6B | yes | 15–30 min |
| 11 | `bash scripts/run_pipeline.sh clef-smoke` | Clef Flash on 20 validation users. **STOP** | no | 2 min |
| 12 | `bash scripts/run_pipeline.sh clef` | Clef Flash on the test eval cohort | no | 20–60 min |
| 13 | `bash scripts/run_pipeline.sh report` | Results table → `results/$CAT/` | no | 1 min |
| 14 | `bash scripts/run_pipeline.sh backup` | Archive again with all results | no | 1 min |

`HL` and `CHANNELS` must be the **same values** for `l1-dryrun` and `l1-freeze`.
Stage 1 needs no `CAT`. Running `bash scripts/run_pipeline.sh` with no stage prints the list.

## 4. What to send back at each STOP

| After | Send |
|---|---|
| screen | the `== Validation screen ==` table |
| l1-tune | the half-life table |
| l1-compare | both tables (valid and test) |
| l1-dryrun | the JSON report (or `data/$CAT/processed/l1_report.json`) |
| clef-smoke | the last 5 lines (users, ndcg, failures, retries, latency, tokens billed) |

Any failure: the last 30 lines of the newest file in `logs/`
(`ls -t logs | head -1`).

## 5. Keep your results

The instance disk is the only copy of the data. **Destroying the instance deletes it.**
Stopping keeps it, but a stopped instance can be impossible to restart if the host is
taken. After freezing and after every model run:

```bash
bash scripts/run_pipeline.sh backup
```

then copy the archive to your laptop (from your laptop's terminal; host and port are on
the instance card in the Vast.ai console):

```bash
scp -P <port> root@<host>:/workspace/rec-l2-bench/backups/<file>.tar.gz .
```

Restore on a new instance with `tar xzf <file>.tar.gz` from the repo root.

## 6. Cost control

- Stop the instance whenever nothing is running; GPU billing stops, storage continues.
- Stages 1, 2, 4–8 and 11–14 do not use the GPU. If you are waiting on a decision,
  stop the instance in between.
