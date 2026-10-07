
"""Day 1 - download WELFake, clean it, fine-tune DistilBERT, save the model.

Run:  python src/train.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from huggingface_hub import hf_hub_download
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, Trainer,
                          TrainingArguments, set_seed)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models" / "distilbert_fake_news"

BASE_MODEL = "distilbert-base-uncased"
MAX_LEN = 256
EPOCHS = 1
BATCH = 8

# Full WELFake: 68,134 train pool / 2,000 val / 2,000 test (72,134 total).
# Lower TRAIN_N to e.g. 20000 for a quick CPU-only run.
TRAIN_N, VAL_N, TEST_N = 68000, 2000, 2000

PARQUET = "data/train-00000-of-00001-290868f0a36350c5.parquet"
HF_REPO = "davanstrien/WELFake"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"[device] Using device: {DEVICE} ({torch.cuda.get_device_name(0)})")
else:
    print(f"[device] Using device: {DEVICE}")


# ---------------------------------------------------------------- 1. data
def download_csv() -> Path:
    """Step 1 - fetch WELFake (72,134 articles) from the HuggingFace mirror of
    the Kaggle CSV, so no Kaggle API token is needed. Cached after first run."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    local = hf_hub_download(HF_REPO, PARQUET, repo_type="dataset")
    out = DATA_DIR / "WELFake.csv"
    if not out.exists():
        df = pd.read_parquet(local)
        df.to_csv(out, index=False, encoding="utf-8")
        print(f"[data] wrote {out} ({len(df):,} rows)")
    return out


def load_frame() -> pd.DataFrame:
    """Step 2 - keep only title/text/label, drop empties, and merge the headline
    into the body so the classifier sees both."""
    print("[data] parsing WELFake.csv (one-time, ~30s) ...", flush=True)
    df = pd.read_csv(download_csv(), low_memory=False)
    df = df[["title", "text", "label"]].dropna(subset=["text", "label"])
    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)

    # only prepend the title when it is not already the first thing in the body
    df["content"] = [
        f"{t} {b}" if t and t.lower() not in b.lower()[:200] else b
        for t, b in zip(df["title"], df["text"])
    ]
    return df


def split(df: pd.DataFrame):
    parts = train_test_split(df, test_size=VAL_N + TEST_N, stratify=df["label"], random_state=42)
    train, rest = parts
    val, test = train_test_split(rest, test_size=TEST_N, stratify=rest["label"], random_state=42)
    train = train.sample(n=min(TRAIN_N, len(train)), random_state=42)
    print(f"[data] train={len(train):,} val={len(val):,} test={len(test):,}"
          f" | fake share={df['label'].mean():.2%}")
    return train, val, test


# ---------------------------------------------------------------- 2. model
def metrics_fn(eval_pred):
    p, y = eval_pred          # EvalPrediction is (predictions, label_ids)
    p = np.argmax(p, axis=1) if np.ndim(p) == 2 else (np.asarray(p) >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y, p),
        "precision": precision_score(y, p, zero_division=0),
        "recall": recall_score(y, p, zero_division=0),
        "f1": f1_score(y, p, zero_division=0),
    }


def main():
    set_seed(42)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train, val, test = split(load_frame())

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=2, id2label={0: "REAL", 1: "FAKE"}, label2id={"REAL": 0, "FAKE": 1}).to(DEVICE)

    def to_ds(frame):
        enc = tok(frame["content"].tolist(), truncation=True, max_length=MAX_LEN)
        return Dataset.from_dict({**enc, "labels": frame["label"].tolist()})

    ds_tr, ds_va, ds_te = map(to_ds, (train, val, test))

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(MODEL_DIR),
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH,
            per_device_eval_batch_size=16,
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup_ratio=0.1,
            eval_strategy="epoch",
            save_strategy="epoch",   # checkpoint before the epoch eval, so a metrics
                                    # bug can't throw away an hour of training
            logging_steps=50,
            report_to=[],
            seed=42,
        ),
        train_dataset=ds_tr,
        eval_dataset=ds_va,
        tokenizer=tok,
        compute_metrics=metrics_fn,
    )
    trainer.train()

    # ---------------------------------------------------------- 3. save
    trainer.save_model(str(MODEL_DIR))   # weights + config (this is what you reload later)
    tok.save_pretrained(str(MODEL_DIR))   # tokenizer must be saved too, or reload breaks
    print(f"[save] model -> {MODEL_DIR}")

    # ---------------------------------------------------------- 4. evaluate
    logits = trainer.predict(ds_te).predictions
    y = ds_te["labels"]
    p = np.argmax(logits, axis=1)
    scores = metrics_fn((logits, y))

    cm = confusion_matrix(y, p)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(cm, cmap="Blues")
    for (r, c), v in np.ndenumerate(cm):
        ax.text(c, r, f"{v:,}", ha="center", va="center")
    ax.set_xticks([0, 1], ["pred REAL", "pred FAKE"])
    ax.set_yticks([0, 1], ["true REAL", "true FAKE"])
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(MODEL_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    out = {
        "model": BASE_MODEL,
        "train_subset": [TRAIN_N, VAL_N, TEST_N],
        "max_len": MAX_LEN,
        "epochs": EPOCHS,
        "test_metrics": {k: round(float(v), 4) for k, v in scores.items()},
        "confusion_matrix": cm.tolist(),
        "report": classification_report(y, p, target_names=["REAL", "FAKE"], output_dict=True),
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(out, indent=2))

    # a few real samples so the demo has one-click examples
    samp = (
        pd.concat([test[test.label == 0].head(2), test[test.label == 1].head(2)])
        [["title", "content", "label"]].to_dict("records"))
    (MODEL_DIR / "samples.json").write_text(json.dumps(samp, indent=2, ensure_ascii=False))

    print("\n=== TEST SET ===")
    print(json.dumps(out["test_metrics"], indent=2))
    print("confusion matrix:", cm.tolist())
    print(classification_report(y, p, target_names=["REAL", "FAKE"], digits=3))


if __name__ == "__main__":
    main()
