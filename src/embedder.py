"""Text embedding and similarity helpers for the resume screener."""

from pathlib import Path

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


def _load_model():
    """Load the embedding model once. Cached by Streamlit when available,
    so repeated Streamlit reruns reuse the same loaded model in memory."""
    try:
        import streamlit as st
        return st.cache_resource(lambda: SentenceTransformer("all-MiniLM-L6-v2"))()
    except ImportError:
        return SentenceTransformer("all-MiniLM-L6-v2")


MODEL = _load_model()


def get_embedding(text: str):
    """Convert a string into a dense embedding vector using MiniLM."""
    return MODEL.encode(text)


def compute_similarity(text1: str, text2: str) -> float:
    """Return cosine similarity of two texts as a float in [0, 1]."""
    embedding1 = get_embedding(text1)
    embedding2 = get_embedding(text2)

    score = cos_sim(embedding1, embedding2).item()

    return float(max(0.0, min(1.0, score)))


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
