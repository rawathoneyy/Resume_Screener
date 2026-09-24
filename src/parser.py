"""PDF text extraction helpers for the resume screener."""

from pathlib import Path
import sys

import pdfplumber


def extract_text_from_pdf(file_path: str) -> str:
    """Open a PDF and return all extractable page text as one string.

    Pages with no text layer (for example scanned images) are skipped
    instead of raising an error.
    """
    # Resolve the path so relative filenames work from any working directory.
    pdf_path = Path(file_path)

    # Collect text from each page that actually has extractable content.
    page_texts: list[str] = []

    # pdfplumber.open() manages the file handle; using it as a context
    # manager guarantees the PDF is closed even if something goes wrong.
    with pdfplumber.open(pdf_path) as pdf:
        # Walk every page in document order so the combined string
        # matches the original resume layout as closely as possible.
        for page in pdf.pages:
            # extract_text() returns None or "" when there is no text layer
            # (common with image-only / scanned pages).
            raw_text = page.extract_text()

            # Skip empty or whitespace-only pages instead of crashing.
            if not raw_text or not raw_text.strip():
                continue

            page_texts.append(raw_text.strip())

    # Join pages with blank lines so section boundaries stay readable.
    return "\n\n".join(page_texts)


if __name__ == "__main__":
    # Quick manual test against a sample resume in the project data folder.
    sample_pdf = (
    Path(__file__).resolve().parent.parent
    / "data/sample_resume/Resume_3_Arjun_Mehta_Data_Analyst.pdf"
)
    extracted = extract_text_from_pdf(sample_pdf)

    # Avoid UnicodeEncodeError on Windows consoles that cannot print PDF bullets.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    print(extracted)
