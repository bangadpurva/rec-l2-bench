# Databricks notebook source
# MAGIC %md
# MAGIC # Category screen (validation only)
# MAGIC Run on a **single-node CPU cluster, 32 GB+ RAM** (no GPU needed). Works from a Git folder
# MAGIC linked to `bangadpurva/rec-l2-bench`. Data goes to the driver's local disk, not the workspace.

# COMMAND ----------

import os
import sys

REPO = os.path.dirname(os.getcwd())          # notebook lives in <repo>/databricks/
sys.path[:0] = [os.path.join(REPO, "src"), os.path.join(REPO, "scripts")]
RAW_ROOT = "/local_disk0/rec-l2-bench/screen"   # driver-local scratch; lost when the cluster stops
os.makedirs(RAW_ROOT, exist_ok=True)
print(REPO, RAW_ROOT)

# COMMAND ----------

# Needs numpy, pandas, pyarrow, pyyaml, scipy: all preinstalled on Databricks Runtime ML/standard.
import screen_categories

table = screen_categories.main([
    "--raw-root", RAW_ROOT,
    "--categories", "Industrial_and_Scientific,Musical_Instruments,Baby_Products,Office_Products",
    "--n-valid", "2000",
])

# COMMAND ----------

cols = ["items", "cut_points", "eligible_test_users", "positives_per_user", "unreachable_share",
        "pop_users_with_pos@100", "cooc_users_with_pos@100", "cooc+pop_users_with_pos@100",
        "cooc+pop_recall@100", "est_test_eval_users"]
display(table[cols].reset_index())
