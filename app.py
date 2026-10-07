"""Day 3 - Streamlit demo.

Run:  streamlit run app.py
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from explain import lime_explain, shap_bar, shap_explain          # noqa: E402
from faithfulness import faithfulness                             # noqa: E402
from predict import MODEL_DIR, predict                            # noqa: E402

st.set_page_config(page_title="Explainable Fake News Detection", layout="wide")

if not (MODEL_DIR / "config.json").exists():
    st.error("No trained model found. Run this first:\n\n    python src/train.py")
    st.stop()

metrics = json.loads((MODEL_DIR / "metrics.json").read_text())
samples = json.loads((MODEL_DIR / "samples.json").read_text())

st.title("Explainable Fake News Detection")
st.caption("DistilBERT classifier + SHAP + LIME, with a perturbation-based faithfulness test. "
           "The model predicts whether an article looks fake based on patterns in WELFake "
           "training data - it does not fact-check claims.")

with st.sidebar:
    st.subheader("Model performance (test set)")
    m = metrics["test_metrics"]
    st.table({"metric": ["accuracy", "precision", "recall", "f1"],
              "value": [f"{m[k]:.2%}" for k in ("accuracy", "precision", "recall", "f1")]})
    st.caption(f"{metrics['train_subset'][0]:,} training articles - "
               f"{metrics['epochs']} epoch(s), max_len {metrics['max_len']}")
    st.image(str(MODEL_DIR / "confusion_matrix.png"), caption="Confusion matrix")
    st.caption("Predicted class from the article's wording and style, not its truth.")


def sample_pick(prefix="main"):
    def load_sample(content):
        st.session_state["text"] = content

    cols = st.columns(len(samples))
    for i, (col, s) in enumerate(zip(cols, samples)):
        label = ("FAKE" if s["label"] else "REAL") + " sample"
        col.button(
            label,
            key=f"{prefix}_sample_{i}",
            on_click=load_sample,
            args=(s["content"],),
            use_container_width=True
        )


draft = st.text_area("News article", key="text", height=160,
                     placeholder="Paste a headline + article here...")
sample_pick()

# st.button is one-shot: every click re-runs this script, so the result must be
# held in session_state. Gating on the button alone makes the whole page vanish
# the moment any later button is pressed.
if st.button("Analyze news", type="primary"):
    for stale in ("shap", "lime", "faith"):
        st.session_state.pop(stale, None)
    st.session_state["analyzed"] = draft.strip()

text = st.session_state.get("analyzed", "")
if len(text.split()) < 5:
    st.stop()

res = predict(text)
col1, col2 = st.columns([1, 1])
col1.metric("Prediction", f"⚠ {res['label']} NEWS" if res["label"] == "FAKE" else f"✔ {res['label']} NEWS")
col2.metric("Confidence", f"{res['confidence']:.1%}")
col1.caption(f"FAKE probability {res['fake_prob']:.1%} | REAL probability {res['real_prob']:.1%}")

cls = res["label_id"]

st.subheader("Important words")
b1, b2 = st.columns(2)
if b1.button("Generate SHAP explanation"):
    with st.spinner("SHAP is masking words and re-scoring..."):
        st.session_state["shap"] = shap_explain(text, cls=cls)
if b2.button("Generate LIME explanation"):
    with st.spinner("LIME is fitting a local surrogate..."):
        st.session_state["lime"] = lime_explain(text, cls=cls)

if "shap" in st.session_state:
    pairs = st.session_state["shap"]
    target = "FAKE" if cls == 1 else "REAL"
    fig = shap_bar(pairs, f"SHAP - contribution to {target} probability")
    st.pyplot(fig)
    plt.close(fig)
    chips = " ".join(
        f"`{w}` {'**+%.2f**' % v if v > 0 else '**%.2f**' % v}" for w, v in pairs[:8])
    st.markdown(chips)
    st.caption(f"Red pushes towards {target}, blue pushes away.")

if "lime" in st.session_state:
    pairs, lime_fig = st.session_state["lime"]
    st.subheader("LIME explanation")
    st.pyplot(lime_fig)
    plt.close(lime_fig)
    st.caption("Local approximation from perturbed copies of this one article.")

st.subheader("Faithfulness test")
st.caption("Delete the words the explanation highlighted, and compare with deleting random words.")
if st.button("Run faithfulness test"):
    with st.spinner("Re-scoring perturbed versions of the article..."):
        st.session_state["faith"] = faithfulness(text, cls=cls)

if "faith" in st.session_state:
    f = st.session_state["faith"]
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Original", f"{f['original']:+.1f}", f"P={f['original_prob']:.1%}")
    b2.metric("Top words removed", f"{f['without_top']:+.1f}", f"{f['drop_top']:+.1f}")
    b3.metric("Random words removed", f"{f['without_random']:+.1f}", f"{f['drop_random']:+.1f}")
    b4.metric("Explanation influence", f["verdict"])
    fig, ax = plt.subplots(figsize=(6, 2.6))
    ax.bar(["original", "top-k removed", "random-k removed"],
           [f["original"], f["without_top"], f["without_random"]],
           color=["#888", "#d64545", "#3b7dd8"])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel(f"log-odds of P({'FAKE' if cls == 1 else 'REAL'})")
    ax.set_title(f"Removing the {f['k']} highlighted words moves the score "
                 f"{abs(f['drop_top'] - f['drop_random']):.1f} more than removing random words")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.markdown(f"Highlighted words: " + " ".join(f"`{w}`" for w in f["words"]))
    st.caption("Score is log-odds, not probability: this model is softmax-saturated, so "
               "probability stays pinned at ~100% while the underlying evidence changes.")
