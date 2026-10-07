"""Day 2/3 - the research component: do the highlighted words actually matter?

Perturbation test (a.k.a. AUF / comprehensiveness):
  1. score the original article
  2. delete the k words with the largest |SHAP| value, rescore
  3. delete k random words (same k, disjoint from step 2), rescore

If the explanation is faithful, step 2 must move the score far more than step 3.
Deletion (not [MASK] substitution) is used because the classifier head was never
trained on the [MASK] embedding - it is not a masked-LM.
"""
import numpy as np

from explain import shap_explain
from predict import MAX_LEN, load


def _offsets(text: str):
    tok, _, _ = load()
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    return enc["input_ids"], enc["offset_mapping"]


def _window(text: str) -> int:
    """Char position where the model's truncated input ends.

    The model only ever reads the first MAX_LEN tokens, so words past this point
    were never seen. Deleting them cannot change the score, which would silently
    dilute the measured gap - so they are excluded from both the top-k and the
    random-k pools."""
    tok, _, _ = load()
    enc = tok(text, truncation=True, max_length=MAX_LEN, return_offsets_mapping=True)
    return max((enc["offset_mapping"][i][1] for i in range(1, len(enc["input_ids"]) - 1)),
               default=0)


def _delete(text: str, spans) -> str:
    for a, b in sorted(spans, reverse=True):
        text = text[:a] + text[b:]
    return " ".join(text.split())


def _margin(text: str, cls: int) -> float:
    """log-odds of cls: log(p_cls / p_other). For 2 classes this is exactly the
    logit margin. Measured instead of probability because a 99%-accurate model is
    softmax-saturated - p sits at 0.99998 and deleting words cannot move it, while
    the margin still moves a lot."""
    p = float(load()[2]([text])[0][cls])
    p = min(max(p, 1e-12), 1 - 1e-12)
    return float(np.log(p / (1 - p)))


def faithfulness(text: str, k: int | None = None, seed: int = 0, cls: int = 1) -> dict:
    _, _, proba = load()
    _, offs = _offsets(text)
    if len(offs) <= 2:
        raise ValueError("article too short to remove words from")

    limit = _window(text)
    usable = [i for i in range(len(offs)) if offs[i][0] < limit]
    usable_set = set(usable)
    if len(usable) <= 2:
        raise ValueError("article's visible part is too short to remove words from")
    if k is None:
        k = max(3, round(0.10 * len(usable)))   # 10% of what the model sees, min 3

    top = [i for i, _ in sorted(
        ((i, abs(v)) for i, (_, v) in enumerate(shap_explain(text, len(offs), cls))),
        key=lambda t: -t[1]) if i in usable_set][:k]

    rng = np.random.default_rng(seed)
    pool = [i for i in usable if i not in set(top)]
    rand = list(rng.choice(pool, size=min(k, len(pool)), replace=False))

    orig = _margin(text, cls)
    drop_top_m = _margin(_delete(text, [offs[i] for i in top]), cls)
    drop_rand_m = _margin(_delete(text, [offs[i] for i in rand]), cls)

    # WordPiece sub-tokens (macron + n) belong to one word: group every hit by the
    # whitespace-delimited word it falls inside, and show that word from the
    # original text rather than the reconstructed token.
    word_of = [text.count(" ", 0, offs[i][0]) for i in range(len(offs))]
    groups = {}
    for i in top:
        groups.setdefault(word_of[i], []).append(i)
    words, seen = [], set()
    for i in top:
        wid = word_of[i]
        if wid in seen:
            continue
        seen.add(wid)
        g = groups[wid]
        span = text[min(offs[j][0] for j in g):max(offs[j][1] for j in g)].strip()
        if span:
            words.append(span)

    drop_top, drop_rand = orig - drop_top_m, orig - drop_rand_m
    rel = abs(drop_top - drop_rand) / max(abs(orig), 1e-6)
    return {
        "k": len(top),
        "words": words,
        "original": orig,
        "without_top": drop_top_m,
        "without_random": drop_rand_m,
        "drop_top": drop_top,
        "drop_random": drop_rand,
        "original_prob": float(proba([text])[0][cls]),
        "faithfulness_score": rel,           # fraction of the original margin
        "verdict": "HIGH" if rel > 1.0 else "MODERATE" if rel > 0.25 else "LOW",
        "text_without_top": _delete(text, [offs[i] for i in top]),
    }
