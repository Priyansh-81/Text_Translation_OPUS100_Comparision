from pathlib import Path
import json, random, time, os
import numpy as np
import torch
import pandas as pd
import sacrebleu

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
RESULTS_DIR = ROOT / "results"
for d in (DATA_DIR, OUTPUT_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def score_translation(references, hypotheses):
    bleu = sacrebleu.corpus_bleu(hypotheses, [references]).score
    chrf = sacrebleu.corpus_chrf(hypotheses, [references]).score
    return round(bleu, 4), round(chrf, 4)

def append_metric(row):
    path = RESULTS_DIR / "metrics.csv"
    df = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        old = old[old["system"] != row["system"]]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False)

def device_name():
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "Apple MPS"
    return "CPU"
