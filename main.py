import os
import re
import logging
from typing import List

import spacy
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import pipeline

# LOGGING
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# LOAD MODELS
logger.info("Loading SpaCy model...")
nlp = spacy.load("en_core_web_sm")

logger.info("Loading BART summarization model...")
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=-1  # CPU
)

# TEXT EXTRACTION
def extract_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def extract_pdf(file_path):
    import PyPDF2
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_docx(file_path):
    import docx
    doc = docx.Document(file_path)
    return "\n".join(p.text for p in doc.paragraphs)

def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        return extract_txt(file_path)
    elif ext == ".pdf":
        return extract_pdf(file_path)
    elif ext == ".docx":
        return extract_docx(file_path)
    else:
        raise Exception("Unsupported file type")

# CLEANING
def clean_and_normalize(text: str) -> str:
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("’", "'").replace("–", "-").replace("—", "-")
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)
    text = re.sub(r"http[s]?://\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# SENTENCE SPLIT
def sentence_split(text: str) -> List[str]:
    doc = nlp(text)
    return [s.text.strip() for s in doc.sents if len(s.text.strip()) > 5]


# CHUNKING

def chunk_text(sentences, chunk_size=800):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=100
    )
    return splitter.split_text(" ".join(sentences))

# SUMMARIZATION

def summarize_chunks(chunks: List[str]) -> str:
    summaries = []

    for chunk in chunks:
        if len(chunk.split()) < 50:
            continue

        try:
            summary = summarizer(
                chunk,
                max_length=150,
                min_length=50,
                do_sample=False
            )[0]["summary_text"]

            summaries.append(summary)

        except Exception as e:
            logger.warning(f"Skipping chunk due to error: {e}")

    return " ".join(summaries)

# EVALUATION (MARKS OUT OF 10)
def evaluate_assignment(text: str) -> int:
    score = 0

    word_count = len(text.split())
    sentence_count = len(sentence_split(text))

    # Length scoring
    if word_count > 300:
        score += 3
    elif word_count > 150:
        score += 2

    # Sentence richness
    if sentence_count > 20:
        score += 2
    elif sentence_count > 10:
        score += 1

    # Structure keywords
    keywords = ["introduction", "conclusion", "result", "analysis", "method"]
    keyword_hits = sum(1 for k in keywords if k in text.lower())
    score += min(keyword_hits, 3)

    # Academic indicators
    if re.search(r"\b(fig|table|reference)\b", text.lower()):
        score += 1

    return min(score, 10)

#  FULL PIPELINE
def evaluate_assignment_pipeline(file_path):
    logger.info("Extracting text...")
    raw_text = extract_text(file_path)

    logger.info("Cleaning text...")
    clean_text = clean_and_normalize(raw_text)

    logger.info("Splitting sentences...")
    sentences = sentence_split(clean_text)

    logger.info("Chunking text...")
    chunks = chunk_text(sentences)

    logger.info("Summarizing...")
    summary = summarize_chunks(chunks)

    logger.info("Evaluating assignment...")
    marks = evaluate_assignment(clean_text)

    return {
        "summary": summary,
        "marks": f"{marks}/10"
    }

# RUN
if __name__ == "__main__":
    file_path = "student_long.pdf"  

    result = evaluate_assignment_pipeline(file_path)

    print("\n📌 SUMMARY:\n")
    print(result["summary"])

    print("\n🎯 EVALUATION SCORE:")
    print(result["marks"])
