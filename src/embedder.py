"""Text embedding and similarity helpers for the resume screener.

Uses Google's Gemini embedding API (gemini-embedding-001) rather than a
locally-run model. This avoids sentence-transformers/transformers/scikit-learn
entirely, after their internal import chains repeatedly hit a Windows
Application Control policy blocking scikit-learn's compiled extensions.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

_client = None


def _get_client():
    """Create the Gemini client once, lazily."""
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Add a line like "
                "GEMINI_API_KEY=your_key to your .env file."
            )
        _client = genai.Client(api_key=api_key)
    return _client


def get_embedding(text: str):
    """Return the embedding vector for a string, via Gemini's API.

    task_type="SEMANTIC_SIMILARITY" tells the model these embeddings will
    be compared for similarity (rather than used for search, classification,
    etc.) - Google's docs specifically recommend this for our use case, and
    without it, unrelated texts scored unexpectedly high (~0.53).
    """
    client = _get_client()
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
    )
    return result.embeddings[0].values


# A calibration baseline: the typical cosine similarity Gemini's embedding
# model produces for two genuinely unrelated pieces of text. Discovered
# empirically (~0.53-0.68 across multiple unrelated-text tests) - this
# model's embedding space is "anisotropic," meaning even unrelated text
# scores moderately high on raw cosine similarity. We rescale against this
# floor so displayed scores are intuitively meaningful (near 0% for
# unrelated content) rather than raw, uncalibrated numbers.
BASELINE_FLOOR = 0.6


def compute_similarity(text1: str, text2: str) -> float:
    """Return a calibrated similarity score in [0, 1].

    Raw cosine similarity from Gemini's embeddings is rescaled against an
    empirically-observed baseline floor (see BASELINE_FLOOR) so that
    unrelated text scores near 0 instead of a misleadingly high raw value.
    """
    embedding1 = get_embedding(text1)
    embedding2 = get_embedding(text2)

    dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
    norm1 = sum(a * a for a in embedding1) ** 0.5
    norm2 = sum(b * b for b in embedding2) ** 0.5

    if norm1 == 0 or norm2 == 0:
        return 0.0

    raw_score = dot_product / (norm1 * norm2)

    # Rescale: raw_score of BASELINE_FLOOR maps to 0, raw_score of 1.0 maps to 1.
    calibrated = (raw_score - BASELINE_FLOOR) / (1 - BASELINE_FLOOR)
    return float(max(0.0, min(1.0, calibrated)))


if __name__ == "__main__":
    from parser import extract_text_from_pdf

    project_root = Path(__file__).resolve().parent.parent
    resume_pdf = (
        project_root / "data/sample_resume/Resume_3_Arjun_Mehta_Data_Analyst.pdf"
    )
    jd_path = project_root / "data/sample_jds/jd_software_developer.txt"

    resume_text = extract_text_from_pdf(resume_pdf)
    job_description = jd_path.read_text(encoding="utf-8")

    score = compute_similarity(resume_text, job_description)
    print(f"Similarity score: {score:.4f}")