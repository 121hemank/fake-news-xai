r"""Headless check that app.py runs, so a demo never dies on a blank Streamlit page.

Run:  .venv\Scripts\python.exe smoke_test.py
"""
import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).parent
sample = json.loads((ROOT / "models" / "distilbert_fake_news" / "samples.json").read_text())[0]


def click(at, label):
    """Click a button by its visible label - indices shift as the page grows."""
    for b in at.button:
        if b.label == label:
            return b.click().run(timeout=600)
    raise AssertionError(f"no button {label!r}; page has {[b.label for b in at.button]}")


def ok(at, step):
    assert not at.exception, f"{step} failed: {at.exception}"
    print(f"{step:<17} OK")


at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=600).run()
ok(at, "startup")

at.text_area[0].set_value(sample["content"])
at = at.run()
ok(at, "text input")

at = click(at, "REAL sample")
ok(at, "sample button")

at = click(at, "Analyze news")
ok(at, "Analyze news")

at = click(at, "Generate SHAP explanation")
ok(at, "SHAP explanation")

at = click(at, "Generate LIME explanation")
ok(at, "LIME explanation")

at = click(at, "Run faithfulness test")
ok(at, "faithfulness")

marked = [m.value for m in at.markdown]
assert any("Highlighted words: `" in m for m in marked), (
    "faithfulness rendered no highlighted words - top-k is empty")
print("highlighted words  OK")

print("\nall UI sections render without error")