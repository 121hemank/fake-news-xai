"""Day 2 - SHAP and LIME explanations for a single article.

SHAP  : model-agnostic attributions from masking words (shap.maskers.Text).
LIME  : local surrogate fitted on ~1500 perturbed copies of the article.
Both report contribution to the FAKE probability.
"""
from functools import lru_cache

import numpy as np
from lime.lime_text import LimeTextExplainer

from predict import load


@lru_cache(maxsize=4)
def _shap_explainer(cls: int):
    """cls=1 explains towards FAKE, cls=0 explains towards REAL."""
    import shap

    tok, _, proba = load()
    return shap.Explainer(lambda texts: proba(texts)[:, cls], shap.maskers.Text(tok))


def shap_explain(text: str, top_k: int = 12, cls: int = 1):
    """-> list of (word, shap_value). Positive pushes towards the target class."""
    sv = _shap_explainer(cls)([text], silent=True)
    pairs = list(zip([str(w) for w in sv[0].data], np.asarray(sv[0].values).ravel()))
    pairs.sort(key=lambda kv: -abs(kv[1]))
    return pairs[:top_k]


@lru_cache(maxsize=1)
def _lime_explainer():
    return LimeTextExplainer(class_names=["REAL", "FAKE"], random_state=42)


def lime_explain(text: str, top_k: int = 10, num_samples: int = 1200, cls: int = 1):
    """-> (list of (word, weight), matplotlib figure).

    Returns a figure, not exp.as_html(): LIME's HTML embeds a webpack bundle that
    Streamlit never executes (unsafe_allow_html does not run <script>), so it would
    render as thousands of lines of raw JavaScript on screen.

    cls must be the predicted class - explaining FAKE while the panel says REAL
    shows the reader contributions to the opposite prediction."""
    _, _, proba = load()

    def row_fn(texts):
        p = proba(texts)[:, 1]
        return np.column_stack([1 - p, p])

    exp = _lime_explainer().explain_instance(
        text, row_fn, labels=(cls,), num_features=top_k, num_samples=num_samples)
    pairs = [(w, float(v)) for w, v in exp.as_list(label=cls)]
    return pairs, exp.as_pyplot_figure(label=cls)


def shap_bar(pairs, title="SHAP - contribution to FAKE probability", width=8):
    import matplotlib.pyplot as plt

    w = [p[0] for p in pairs][::-1]
    v = [p[1] for p in pairs][::-1]
    fig, ax = plt.subplots(figsize=(width, max(2.2, 0.32 * len(w))))
    ax.barh(w, v, color=["#d64545" if x > 0 else "#3b7dd8" for x in v])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title(title, fontsize=11)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    return fig
