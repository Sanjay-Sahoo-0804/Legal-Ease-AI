import os
from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_text_from_pdf(pdf_path: str) -> str:
    """Reads a PDF file and extracts all its raw text content page by page."""
    print(f"🔄 Extracting text from: {pdf_path}")
    try:
        reader = PdfReader(pdf_path)
        full_text = []

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                full_text.append(text)

        raw_content = "\n".join(full_text)
        print(f"✅ Extraction complete. Character count: {len(raw_content)}")
        return raw_content

    except Exception as e:
        print(f"❌ Error reading PDF {pdf_path}: {str(e)}")
        return ""


def clean_text(raw_text: str) -> str:
    """Performs basic cleaning to remove extra whitespace and artifacts."""
    # Split text into lines, strip whitespaces, and remove completely empty lines
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    return "\n".join(lines)


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    """Splits raw text into mathematically manageable chunks with overlapping bounds.

    - chunk_size: Max characters per chunk.
    - chunk_overlap: Characters shared between consecutive chunks to keep context.
    """
    print(f"✂️ Chunking text (Size: {chunk_size}, Overlap: {chunk_overlap})...")

    # Separators listed by priority: paragraphs (\n\n), sentences (\n, .), then spaces
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = text_splitter.split_text(text)
    print(f"✅ Created {len(chunks)} text chunks.")
    return chunks


if __name__ == "__main__":
    # Sample execution path to test your script locally
    # Create a dummy data directory for testing if it doesn't exist
    sample_data_dir = Path("sample_data")
    sample_data_dir.mkdir(exist_ok=True)

    print(
        f"\n🚀 Preprocessing Pipeline Ready! Place your raw PDFs inside the '{sample_data_dir}' folder."
    )
    print(
        "To test this script, pass a valid PDF path to 'extract_text_from_pdf()'."
    )
