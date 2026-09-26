"""Embedding-based semantic similarity between a resume and a job description."""

import os
from pathlib import Path

import cohere
from dotenv import load_dotenv

load_dotenv()

_client = None

# Empirically-observed baseline: raw cosine similarity between two totally
# UNRELATED texts, using Cohere's embed-english-v3.0. Without subtracting
# this floor, unrelated texts score misleadingly high instead of near 0.
# IMPORTANT: this value is a placeholder — re-measure it for Cohere as
# described in the deployment steps, then update this number.
BASELINE_FLOOR = 0.14


def _get_api_key(key_name: str) -> str | None:
    """Look for a secret first in Streamlit Cloud's st.secrets (used when
    deployed), falling back to a plain environment variable / .env file
    (used for local development)."""
    try:
        import streamlit as st
        if key_name in st.secrets:
            return st.secrets[key_name]
    except Exception:
        pass
    return os.environ.get(key_name)


def _get_client():
    global _client
    if _client is None:
        api_key = _get_api_key("COHERE_API_KEY")
        if not api_key:
            raise ValueError(
                "COHERE_API_KEY not found. Add it to .env locally, or to "
                "Streamlit Cloud's Secrets settings when deployed."
            )
        _client = cohere.Client(api_key)
    return _client


def get_embedding(text: str) -> list[float]:
    """Return the embedding vector for a single piece of text."""
    client = _get_client()
    response = client.embed(
        texts=[text],
        model="embed-english-v3.0",
        input_type="search_document",
    )
    return response.embeddings[0]


def compute_similarity(text1: str, text2: str) -> float:
    """Return a 0-1 similarity score between two texts, calibrated against an
    empirically-observed baseline floor (see BASELINE_FLOOR) so that
    unrelated text scores near 0 instead of a misleadingly high raw value.
    """
    embedding1 = get_embedding(text1)
    embedding2 = get_embedding(text2)

    dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
    norm1 = sum(a * a for a in embedding1) ** 0.5
    norm2 = sum(b * b for b in embedding2) ** 0.5
    raw_similarity = dot_product / (norm1 * norm2)

    calibrated = (raw_similarity - BASELINE_FLOOR) / (1 - BASELINE_FLOOR)
    return max(0.0, min(1.0, calibrated))

if __name__ == "__main__":
    from parser import extract_text_from_pdf

    project_root = Path(__file__).resolve().parent.parent
    resume_pdf = (
        project_root / "data/sample_resume/Resume_1_Rahul_Sharma_Software_Developer.pdf"
    )
    jd_path = project_root / "data/sample_jds/jd_software_developer.txt"

    resume_text = extract_text_from_pdf(resume_pdf)
    job_description = jd_path.read_text(encoding="utf-8")

    score = compute_similarity(resume_text, job_description)
    print(f"Similarity score: {score:.4f}")