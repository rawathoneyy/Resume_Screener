"""LLM-powered explanation of a resume-JD match, using Groq's API."""

import os

from dotenv import load_dotenv
from groq import Groq

# Load variables from .env into the environment (only affects this process).
load_dotenv()

_client = None


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
        api_key = _get_api_key("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Add it to .env locally, or to "
                "Streamlit Cloud's Secrets settings when deployed."
            )
        _client = Groq(api_key=api_key)
    return _client

def explain_match(score: float, missing_skills: list[str]) -> str:
    """Ask an LLM to explain a match score and missing skills in plain
    English. The LLM only narrates data we already computed ourselves —
    it does not recalculate the score or re-derive the skill list."""
    client = _get_client()

    missing_text = ", ".join(missing_skills) if missing_skills else "none identified"

    system_prompt = (
        "You are helping explain resume-to-job-description match results "
        "to a job seeker. You will be given a match score (0-100%) and a "
        "list of missing skills that were already calculated by other "
        "logic. Write a short, constructive, 2-3 sentence explanation of "
        "what this means for the candidate. Only reference the score and "
        "skills given to you - do not invent additional skills, "
        "qualifications, or details that were not provided."
    )

    user_prompt = (
        f"Match score: {score * 100:.0f}%\n"
        f"Missing skills: {missing_text}\n\n"
        "Write the explanation now."
    )

    response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
    temperature=0.4,
    max_tokens=400,
    reasoning_effort="low",
)

    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    # Quick manual test using numbers we already know from earlier phases.
    test_score = 0.7177
    test_missing = ["Agile", "Database Design", "MySQL", "Problem Solving"]

    explanation = explain_match(test_score, test_missing)
    print("Explanation:")
    print(explanation)