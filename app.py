"""Streamlit frontend for the AI Resume Screener."""

import sys
from pathlib import Path

import streamlit as st

# Allow importing from src/ regardless of the folder streamlit is launched from.
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from parser import extract_text_from_pdf
from keyword_analysis import find_missing_keywords

st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="📄",
    layout="wide",
)

# ---- Custom styling: overrides Streamlit's default look ----
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2.5rem;
        max-width: 900px;
    }
    .metric-card {
        background-color: #f6f8fa;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        text-align: center;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #666;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 600;
        color: #1a1a1a;
    }
    .skill-tag {
        display: inline-block;
        background-color: #fff4e5;
        color: #b3691d;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 0.85rem;
        margin: 4px 6px 4px 0;
    }
    .matched-tag {
        display: inline-block;
        background-color: #e6f4ea;
        color: #1e7b34;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 0.85rem;
        margin: 4px 6px 4px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- Header ----
st.markdown("### 📄 AI Resume Screener")
st.caption("Match a resume against a job description and see what's missing.")

# ---- Inputs, side by side ----
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Resume (PDF)**")
    uploaded_resume = st.file_uploader(
        "Upload resume", type=["pdf"], label_visibility="collapsed"
    )

with col2:
    st.markdown("**Job description**")
    jd_text_input = st.text_area(
        "Paste job description",
        height=150,
        label_visibility="collapsed",
        placeholder="Paste the job description here...",
    )

analyze_clicked = st.button("Analyze match", type="primary", use_container_width=True)

st.divider()

# ---- Results ----
if analyze_clicked:
    # Validate inputs before doing any real work.
    if uploaded_resume is None:
        st.warning("Upload a resume PDF before analyzing.")
        st.stop()
    if not jd_text_input or not jd_text_input.strip():
        st.warning("Paste a job description before analyzing.")
        st.stop()

    with st.spinner("Analyzing..."):
        from embedder import compute_similarity  # loaded only when needed

        resume_text = extract_text_from_pdf(uploaded_resume)

        if not resume_text.strip():
            st.error(
                "Couldn't extract any text from this PDF. It may be a "
                "scanned image without a real text layer."
            )
            st.stop()

        score = compute_similarity(resume_text, jd_text_input)
        missing_skills = find_missing_keywords(resume_text, jd_text_input)

    score_col, missing_col = st.columns(2)
    with score_col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Match score</div>
                <div class="metric-value">{score * 100:.0f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with missing_col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Missing skills found</div>
                <div class="metric-value">{len(missing_skills)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown("**Missing skills**")
    if missing_skills:
        tags_html = "".join(
            f'<span class="skill-tag">{skill}</span>' for skill in missing_skills
        )
        st.markdown(tags_html, unsafe_allow_html=True)
    else:
        st.markdown(
            '<span class="matched-tag">No obvious gaps found</span>',
            unsafe_allow_html=True,
        )
else:
    st.info("Upload a resume and paste a job description, then click Analyze match.")
