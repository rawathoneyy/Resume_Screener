"""Text embedding and similarity helpers for the resume screener."""

from pathlib import Path

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# Load the model once when this module is imported, not inside get_embedding().
# SentenceTransformer initialization downloads weights (first time) and moves
# them onto CPU/GPU; repeating that on every call would dominate runtime.
MODEL = SentenceTransformer("all-MiniLM-L6-v2")


def get_embedding(text: str):
    """Convert a string into a dense embedding vector using MiniLM."""
    return MODEL.encode(text)


def compute_similarity(text1: str, text2: str) -> float:
    """Return cosine similarity of two texts as a float in [0, 1]."""
    embedding1 = get_embedding(text1)
    embedding2 = get_embedding(text2)

    # cos_sim is PyTorch-based and returns a 1x1 tensor of cosine scores.
    score = cos_sim(embedding1, embedding2).item()

    # Clip so tiny negative cosine values from floating-point noise stay in [0, 1].
    return float(max(0.0, min(1.0, score)))


if __name__ == "__main__":
    from parser import extract_text_from_pdf

    project_root = Path(__file__).resolve().parent.parent
    resume_pdf = (
        project_root
        / "data/sample_resume/Resume_3_Arjun_Mehta_Data_Analyst.pdf"
    )
    jd_path = project_root / "data/sample_jds/jd_software_developer.txt"

    resume_text = extract_text_from_pdf(resume_pdf)
    job_description = jd_path.read_text(encoding="utf-8")

    score = compute_similarity(resume_text, job_description)
    print(f"Similarity score: {score:.4f}")
