"""Keyword gap analysis between a resume and a job description."""

import re
from pathlib import Path

# Curated skills for software/tech screening. Kept as a set so membership
# checks are O(1) and duplicate entries cannot sneak in.
CURATED_SKILLS = {
    "Python",
    "Java",
    "JavaScript",
    "TypeScript",
    "SQL",
    "Django",
    "Flask",
    "FastAPI",
    "REST API",
    "Git",
    "HTML",
    "CSS",
    "PostgreSQL",
    "MySQL",
    "MongoDB",
    "Docker",
    "Kubernetes",
    "Linux",
    "AWS",
    "React",
    "Node.js",
    "Object-Oriented Programming",
    "Unit Testing",
    "Agile",
    "Debugging",
    "API Development",
    "Database Design",
    "Version Control",
    "CI/CD",
    "Pandas",
    "NumPy",
    "Machine Learning",
    "Communication",
    "Problem Solving",
    "Teamwork",
    "Collaboration",
    "Leadership",
    "Time Management",
    "Attention to Detail",
    "Documentation",
    "Code Review",
}


def _skill_variants(skill: str) -> set[str]:
    """Build lowercase spellings that should count as the same skill."""
    lower = skill.lower()
    variants = {lower, lower.replace("-", " "), lower.replace(" ", "-")}

    # Common plural / wording differences seen in JDs and resumes.
    if lower == "rest api":
        variants.update({"rest apis", "restful api", "restful apis"})
    elif lower == "problem solving":
        variants.update({"problem-solving", "problem solving skills"})
    elif lower == "object-oriented programming":
        variants.update({"object oriented programming", "oop"})
    elif lower == "unit testing":
        variants.update({"unit tests", "unit test"})
    elif lower == "version control":
        variants.update({"source control"})
    elif lower == "code review":
        variants.update({"code reviews"})
    elif lower == "api development":
        variants.update({"api development", "developing apis"})
    elif lower == "ci/cd":
        variants.update({"ci/cd", "ci-cd", "continuous integration"})

    return variants


def _skill_in_text(skill: str, text: str) -> bool:
    """Return True if the skill appears in text (case-insensitive)."""
    haystack = text.lower()

    # PDFs sometimes hyphenate a word right at a line break (e.g. "Object-\nOriented").
    # Join those back together first, keeping the hyphen, before treating any
    # remaining newlines as plain spaces.
    haystack = re.sub(r"-\s*\n\s*", "-", haystack)
    haystack = haystack.replace("\n", " ")

    for variant in _skill_variants(skill):
        if " " in variant or "/" in variant:
            if variant in haystack:
                return True
            continue
        if re.search(rf"\b{re.escape(variant)}\b", haystack):
            return True
    return False


def find_missing_keywords(resume_text: str, jd_text: str) -> list[str]:
    """Return curated skills that the JD asks for but the resume never mentions."""
    # Only skills the employer actually listed (or paraphrased) in the JD.
    jd_skills = {skill for skill in CURATED_SKILLS if _skill_in_text(skill, jd_text)}

    # Among those, keep skills that never appear in the resume.
    missing = [skill for skill in jd_skills if not _skill_in_text(skill, resume_text)]

    return sorted(missing)


if __name__ == "__main__":
    from parser import extract_text_from_pdf

    project_root = Path(__file__).resolve().parent.parent
    resume_pdf = (
        project_root / "data/sample_resume/Resume_1_Rahul_Sharma_Software_Developer.pdf"
    )
    jd_path = project_root / "data/sample_jds/jd_software_developer.txt"

    resume_text = extract_text_from_pdf(resume_pdf)
    job_description = jd_path.read_text(encoding="utf-8")

    missing_skills = find_missing_keywords(resume_text, job_description)
    print("Missing skills:")
    for skill in missing_skills:
        print(f"- {skill}")
