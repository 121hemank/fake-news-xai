"""Terminal run of the full pipeline on the bundled samples - no UI needed."""
import json
from pathlib import Path

from explain import lime_explain, shap_explain
from faithfulness import faithfulness
from predict import MODEL_DIR, predict

samples = json.loads((MODEL_DIR / "samples.json").read_text())

for s in samples:
    text = s["content"]
    r = predict(text)
    print("=" * 70)
    print(f"[truth {s['label']}]  model says {r['label']} @ {r['confidence']:.1%}")
    print(f"  {text[:150]}...")

    shap = shap_explain(text, cls=r["label_id"])
    print("  SHAP  " + "  ".join(f"{w}:{v:+.2f}" for w, v in shap[:6]))

    lime, _ = lime_explain(text)
    print("  LIME  " + "  ".join(f"{w}:{v:+.2f}" for w, v in lime[:6]))

    f = faithfulness(text, cls=r["label_id"])
    print(f"  FAITH original={f['original']:.1%}  top-{f['k']}-removed={f['without_top']:.1%}  "
          f"random-removed={f['without_random']:.1%}  gap={f['faithfulness_score']:+.1%} "
          f"[{f['verdict']}]")
    print(f"         words: {', '.join(f['words'])}")
