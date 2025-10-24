#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parse a natural-language request like:
  "Run BLCA with SVM using mRNA+miRNA+SNV, 10-fold CV. English report please."
and write /tmp/DeepKEGG-agent/last_run_config.json
"""
import json, re, os, sys

BASE = "/tmp/DeepKEGG-master"
OUT  = "/tmp/DeepKEGG-agent/last_run_config.json"

# ---- read NL input ----
if len(sys.argv) > 1:
    nl = " ".join(sys.argv[1:])
else:
    try:
        nl = input("Enter your request (e.g. 'Run BLCA with SVM using mRNA+miRNA+SNV, 10-fold CV. English report please.'): ").strip()
    except EOFError:
        nl = ""

if not nl:
    print("No input. Example:\n  Run BLCA with SVM using mRNA+miRNA+SNV, 10-fold CV. English report please.")
    sys.exit(1)

txt = nl.lower()

# ---- dictionaries ----
cancers = ["lihc","blca","brca","prad","aml","wt"]
models_map = {
    "svm": "经典ML：SVM",
    "xgb": "经典ML：XGBoost",
    "xgboost": "经典ML：XGBoost",
    "lr": "经典ML：LR",
    "logistic": "经典ML：LR",
    "deepkegg": "DeepKEGG（论文原法）",
}

omics_all = ["mRNA","miRNA","SNV"]

# ---- parse cancer ----
cancer = None
for c in cancers:
    if re.search(rf"\b{c}\b", txt):
        cancer = c.upper()
        break
if cancer is None:
    print("[warn] Cancer not found from NL. Defaulting to LIHC.")
    cancer = "LIHC"

# ---- parse model ----
model = None
for k,v in models_map.items():
    if re.search(rf"\b{k}\b", txt):
        model = v
        break
if model is None:
    print("[warn] Model not found from NL. Defaulting to 经典ML：SVM.")
    model = "经典ML：SVM"

# ---- parse omics ----
omics = []
if "all omics" in txt or "all modalities" in txt or "all data" in txt:
    omics = omics_all[:]
else:
    if "mrna" in txt:  omics.append("mRNA")
    if "mirna" in txt: omics.append("miRNA")
    if re.search(r"\bsnv\b", txt): omics.append("SNV")
if not omics:
    print("[warn] Omics not found from NL. Defaulting to mRNA+miRNA+SNV.")
    omics = omics_all[:]

# ---- parse CV ----
# patterns: "10-fold", "10 fold", "cv=10", repeats like "repeat 2" or "2x1"
cv_folds  = 5
cv_repeats= 1
m = re.search(r"(\d+)\s*-\s*fold|\b(\d+)\s*fold|\bcv\s*=\s*(\d+)", txt)
if m:
    val = next(int(g) for g in m.groups() if g)
    cv_folds = val

m = re.search(r"repeat\s+(\d+)|(\d+)\s*x\s*(\d+)", txt)
if m:
    nums = [int(g) for g in m.groups() if g]
    if len(nums) == 1:
        cv_repeats = nums[0]
    elif len(nums) >= 2:
        cv_folds, cv_repeats = nums[0], nums[1]

# ---- parse language ----
report_lang = "en"
if "chinese" in txt or "中文" in txt or "zh" in txt:
    report_lang = "zh"
if "english" in txt or "英文" in txt or "en" in txt:
    report_lang = "en"

# ---- parse k (optional) ----
k = 64
m = re.search(r"\bk\s*=\s*(\d+)", txt)
if m:
    k = int(m.group(1))

# ---- build paths by cancer ----
data_dir = f"{BASE}/{cancer}_data"
paths = {
    "clinical": f"{data_dir}/response.csv",
    "mrna":     f"{data_dir}/mRNA_data.csv",
    "mirna":    f"{data_dir}/miRNA_data.csv",
    "snv":      f"{data_dir}/snv_data.csv",
    "kegg_gmt": f"{BASE}/KEGG_pathways/20230205_kegg_hsa.gmt",
    "kegg_map_long": f"{BASE}/KEGG_pathways/kegg_anano.txt",
    "id_column": "index",
    "label_column": "response",
}

cfg = {
    "data_source": "manifest-nl",
    "nl_request":  nl,
    "report_lang": report_lang,
    "cancer": cancer,
    "omics": omics,
    "model": model,
    "cv_folds": cv_folds,
    "cv_repeats": cv_repeats,
    "k": k,
    "paths": paths,
    "output_root": "/tmp/DeepKEGG-agent/runs",
}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=2)

print("\n===== Parsed from Natural Language =====")
print(json.dumps(cfg, ensure_ascii=False, indent=2))
print("\nSaved to:", OUT)
print("Next:\n  python /tmp/DeepKEGG-agent/step3_run.py\n  python /tmp/DeepKEGG-agent/step5_report.py")
