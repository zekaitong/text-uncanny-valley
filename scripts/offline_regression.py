#!/usr/bin/env python3
"""
Offline regression checks for core pipeline logic.

No API calls. Uses small synthetic fixtures to validate:
- omission/NIAH scoring in tests/test_llms.py
- summarize_f1 collection + output schema
- NIAH recompute and summary-build utilities
- Experiment C answer parsing modes
"""

import json
import os
import tempfile
from pathlib import Path
import importlib.util
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))


def import_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


test_llms = import_from_path("test_llms_module", ROOT / "tests" / "test_llms.py")
summarize_f1 = import_from_path("summarize_f1_module", ROOT / "summaries" / "summarize_f1.py")
recompute_niah = import_from_path("recompute_niah_module", ROOT / "summaries" / "recompute_niah_scores.py")
build_niah = import_from_path("build_niah_module", ROOT / "summaries" / "build_niah_summary_from_results.py")
math_test = import_from_path("math_test_module", ROOT / "gsm8k_test" / "math_test.py")


def assert_close(a, b, eps=1e-9):
    if abs(a - b) > eps:
        raise AssertionError(f"{a} != {b}")


def test_evaluate_response_omission():
    doc = {
        "original_context": "a\nb\nc",
        "omitted_index": [1],
    }
    # model returns exactly omitted line b
    result = test_llms.evaluate_response(["b", 0, ""], doc, use_needle=False, delimiter="\n")
    assert result["tp"] == 1 and result["fp"] == 0 and result["fn"] == 0
    assert_close(result["micro_f1"], 1.0)


def test_evaluate_response_niah_strict_and_hallucination_penalty():
    doc = {
        "original_context": "x\ny\nz",
        "omitted_index": [],
        "needles": ["needle one", "needle two"],
        "needle_depths": [0.3, 0.7],
    }
    # includes one true needle + one hallucination line
    response = "needle one\nGot it! Here are the extra lines:\n"
    result = test_llms.evaluate_response([response, 0, ""], doc, use_needle=True, delimiter="\n")
    assert result["tp"] == 1
    assert result["fn"] == 1
    assert result["fp"] == 1
    assert "Got it! Here are the extra lines:" in result["wrongly_identified_lines"]


def test_summarize_f1_collect():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for cond, score in [("ori", 0.9), ("p_0_1", 0.8)]:
            d = root / cond
            d.mkdir(parents=True, exist_ok=True)
            payload = {
                "test_date": "x",
                "openai": {"gpt-5.2": {"average_accuracy": score}},
            }
            (d / "results.json").write_text(json.dumps(payload), encoding="utf-8")

        rates, series, skipped_u, skipped_m, skipped_b = summarize_f1.collect(str(root))
        assert rates == [0.0, 0.1]
        assert "openai:gpt-5.2" in series
        assert series["openai:gpt-5.2"] == [0.9, 0.8]
        assert not skipped_u and not skipped_m and not skipped_b


def test_recompute_niah_doc():
    doc = {
        "original_context": "l1\nl2",
        "needles": ["n1", "n2"],
        "needle_depths": [0.2, 0.8],
    }
    res = recompute_niah.recompute_niah_for_doc(doc, "n1\nhallucination\n")
    assert res["tp"] == 1 and res["fn"] == 1 and res["fp"] == 1
    assert "hallucination" in res["wrongly_identified_lines"]


def test_build_niah_summary():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        # layout: root/ld/modelA/wa_bn/{ori,1}/results.fixed2.json
        for cond, score in [("ori", 0.5), ("1", 0.4)]:
            p = root / "ld" / "modelA" / "wa_bn" / cond
            p.mkdir(parents=True, exist_ok=True)
            payload = {"test_date": "x", "openai": {"gpt-5.2": {"average_accuracy": score}}}
            (p / "results.fixed2.json").write_text(json.dumps(payload), encoding="utf-8")

        out = build_niah.build_domain_summary(root / "ld", "results.fixed2.json")
        assert out["rates"] == [0.0, 0.1]
        assert out["series"]["openai:gpt-5.2"] == [0.5, 0.4]


def test_math_parsers():
    pred, method = math_test.extract_answer_strict("... #### 42")
    assert pred == "42" and method == "hash"
    pred2, method2 = math_test.extract_answer_strict("The answer is 42")
    assert pred2 is None and method2 == "none"
    pred3, method3 = math_test.extract_answer_legacy("The answer is 42")
    assert pred3 == "42" and method3 == "answer_phrase"


def main():
    tests = [
        test_evaluate_response_omission,
        test_evaluate_response_niah_strict_and_hallucination_penalty,
        test_summarize_f1_collect,
        test_recompute_niah_doc,
        test_build_niah_summary,
        test_math_parsers,
    ]
    for t in tests:
        t()
        print(f"[PASS] {t.__name__}")
    print(f"All offline regression checks passed ({len(tests)} tests).")


if __name__ == "__main__":
    main()
