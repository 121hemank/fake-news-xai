"""Load the saved model once and turn text into Fake/Real + confidence.

Every other module imports from here, and `python src/predict.py "<text>"`
is a quick sanity check that the saved model works.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "distilbert_fake_news"
MAX_LEN = 256   # must match train.py, else inference sees longer context than training
LABELS = {0: "REAL", 1: "FAKE"}


@lru_cache(maxsize=1)
def load():
    """Returns (tokenizer, model, batched predict_proba). Cached per process."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device).eval()

    def predict_proba(texts) -> np.ndarray:
        """texts: list[str] -> (n, 2) array of [real_prob, fake_prob]."""
        texts = [t if isinstance(t, str) and t.strip() else " " for t in texts]
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), 16):
                batch = tok(texts[i:i + 16], return_tensors="pt", padding=True,
                            truncation=True, max_length=MAX_LEN).to(device)
                out.append(torch.softmax(model(**batch).logits, dim=-1).cpu().numpy())
        return np.concatenate(out)

    return tok, model, predict_proba


@lru_cache(maxsize=1)
def classifier():
    """Plain HF pipeline, for one-off calls outside the app."""
    tok, model, _ = load()
    return pipeline("text-classification", model=model, tokenizer=tok,
                    truncation=True, max_length=MAX_LEN)


def predict(text: str) -> dict:
    _, _, proba = load()
    p = proba([text])[0]
    idx = int(p.argmax())
    return {
        "label": LABELS[idx],
        "label_id": idx,
        "confidence": float(p[idx]),
        "fake_prob": float(p[1]),
        "real_prob": float(p[0]),
    }


if __name__ == "__main__":
    import sys

    assert MODEL_DIR.exists(), f"no trained model at {MODEL_DIR} - run: python src/train.py"
    demo = sys.argv[1] if len(sys.argv) > 1 else "SHOCKING! Scientists have discovered that chocolate cures all disease."
    r = predict(demo)
    print(f"{r['label']}  {r['confidence']:.1%}   (fake {r['fake_prob']:.3f} / real {r['real_prob']:.3f})")
