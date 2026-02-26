import fitz  # PyMuPDF
import re
import pickle
from tqdm import tqdm

PDF_PATH = "data/handbook.pdf"
OUTPUT_CHUNKS = "chunks.pkl"

CHUNK_SIZE = 500 
OVERLAP = 100


def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\n', ' ')
    return text.strip()


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""

    for page in doc:
        full_text += page.get_text()

    return clean_text(full_text)


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


if __name__ == "__main__":
    print("Extracting text from PDF...")
    text = extract_text_from_pdf(PDF_PATH)

    print("Chunking text...")
    chunks = chunk_text(text)

    print(f"Total chunks created: {len(chunks)}")

    with open(OUTPUT_CHUNKS, "wb") as f:
        pickle.dump(chunks, f)

    print("Chunks saved to chunks.pkl")