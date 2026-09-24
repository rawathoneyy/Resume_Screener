"""Sanity checks for compute_similarity (plain asserts, no pytest)."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from embedder import compute_similarity
from parser import extract_text_from_pdf

RESUME_1 = (
    PROJECT_ROOT
    / "data/sample_resume/Resume_1_Rahul_Sharma_Software_Developer.pdf"
)
RESUME_2 = (
    PROJECT_ROOT
    / "data/sample_resume/Resume_2_Priya_Verma_Customer_Service.pdf"
)
SOFTWARE_JD = PROJECT_ROOT / "data/sample_jds/jd_software_developer.txt"


def test_nearly_identical_sentences_score_high():
    text1 = "The software developer builds REST APIs with Python and Django."
    text2 = "The software developer builds REST APIs using Python and Django."
    score = compute_similarity(text1, text2)
    assert score > 0.8, f"expected score > 0.8 for similar sentences, got {score:.4f}"
    return score


def test_unrelated_sentences_score_low():
    text1 = "Python Django REST APIs and PostgreSQL database design."
    text2 = "A chocolate cake recipe with flour, sugar, and cocoa powder."
    score = compute_similarity(text1, text2)
    assert score < 0.3, f"expected score < 0.3 for unrelated sentences, got {score:.4f}"
    return score


def test_resume_1_matches_software_jd_better_than_resume_2():
    resume_1_text = extract_text_from_pdf(RESUME_1)
    resume_2_text = extract_text_from_pdf(RESUME_2)
    job_description = SOFTWARE_JD.read_text(encoding="utf-8")

    score_1 = compute_similarity(resume_1_text, job_description)
    score_2 = compute_similarity(resume_2_text, job_description)
    gap = score_1 - score_2

    # Resume 1 is a software developer; Resume 2 is customer service.
    # A 0.1 gap is large enough to count as a real ranking difference.
    assert score_1 > score_2, (
        f"expected Resume 1 ({score_1:.4f}) > Resume 2 ({score_2:.4f})"
    )
    assert gap > 0.1, (
        f"expected Resume 1 to beat Resume 2 by more than 0.1, gap was {gap:.4f}"
    )
    return score_1, score_2, gap


def run_test(name, fn):
    try:
        result = fn()
        print(f"PASS: {name} - {result}")
        return True
    except AssertionError as exc:
        print(f"FAIL: {name} - {exc}")
        return False
    except Exception as exc:
        print(f"FAIL: {name} - unexpected error: {exc}")
        return False


if __name__ == "__main__":
    results = [
        run_test(
            "nearly identical sentences score above 0.8",
            test_nearly_identical_sentences_score_high,
        ),
        run_test(
            "unrelated sentences score below 0.3",
            test_unrelated_sentences_score_low,
        ),
        run_test(
            "Resume 1 scores meaningfully higher than Resume 2 on the software JD",
            test_resume_1_matches_software_jd_better_than_resume_2,
        ),
    ]

    passed = sum(results)
    failed = len(results) - passed
    print()
    print(f"Summary: {passed} passed, {failed} failed, {len(results)} total")
    if failed:
        sys.exit(1)
