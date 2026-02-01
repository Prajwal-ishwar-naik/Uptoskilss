import os
import re
import logging
from typing import List
import io

import spacy
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import pipeline
from fastapi import FastAPI, UploadFile, File

import PyPDF2
import docx

# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- LOAD MODELS ----------------
logger.info("Loading SpaCy model...")
nlp = spacy.load("en_core_web_sm")

logger.info("Loading summarization model...")
summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    device=-1
)

# ---------------- FASTAPI INIT ----------------
app = FastAPI(title="Document Summarizer API")

# ---------------- TEXT EXTRACTION ----------------
def extract_txt(file_bytes):
    return file_bytes.decode("utf-8", errors="ignore")

def extract_pdf(file_bytes):
    text = ""
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def extract_docx(file_bytes):
    doc = docx.Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs)

def extract_text(file_bytes, filename):
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".txt":
        return extract_txt(file_bytes)
    elif ext == ".pdf":
        return extract_pdf(file_bytes)
    elif ext == ".docx":
        return extract_docx(file_bytes)
    else:
        raise Exception("Unsupported file type")

# ---------------- CLEANING ----------------
def clean_and_normalize(text: str) -> str:
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("’", "'").replace("–", "-").replace("—", "-")
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)
    text = re.sub(r"http[s]?://\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ---------------- SENTENCE SPLIT ----------------
def sentence_split(text: str) -> List[str]:
    doc = nlp(text)
    return [s.text.strip() for s in doc.sents if len(s.text.strip()) > 5]

# ---------------- CHUNKING ----------------
def chunk_text(sentences, chunk_size=600):  # reduced slightly for safety
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=100
    )
    return splitter.split_text(" ".join(sentences))

# ---------------- SUMMARIZATION ----------------
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

    if not summaries:
        return "Text too short to generate a proper summary."

    return " ".join(summaries)

# ---------------- EVALUATION ----------------
def evaluate_assignment(text: str) -> int:
    score = 0
    word_count = len(text.split())
    sentence_count = len(sentence_split(text))

    if word_count > 300:
        score += 3
    elif word_count > 150:
        score += 2

    if sentence_count > 20:
        score += 2
    elif sentence_count > 10:
        score += 1

    keywords = ["introduction", "conclusion", "result", "analysis", "method"]
    keyword_hits = sum(1 for k in keywords if k in text.lower())
    score += min(keyword_hits, 3)

    if re.search(r"\b(fig|table|reference)\b", text.lower()):
        score += 1

    return min(score, 10)

# ---------------- SAVE OUTPUT ----------------
def save_summary_to_txt(filename, summary, marks):
    os.makedirs("outputs", exist_ok=True)

    safe_name = filename.replace(" ", "_")
    output_file = os.path.join("outputs", f"result_{safe_name}.txt")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("===== DOCUMENT SUMMARY =====\n\n")
        f.write(summary + "\n\n")
        f.write("===== EVALUATION SCORE =====\n")
        f.write(marks)

    return output_file

# ---------------- FULL PIPELINE ----------------
def process_file(file_bytes, filename):
    raw_text = extract_text(file_bytes, filename)
    clean_text = clean_and_normalize(raw_text)
    sentences = sentence_split(clean_text)
    chunks = chunk_text(sentences)
    summary = summarize_chunks(chunks)
    marks = f"{evaluate_assignment(clean_text)}/10"
    saved_file = save_summary_to_txt(filename, summary, marks)
    return summary, marks, saved_file

# ---------------- FASTAPI ENDPOINT ----------------
@app.post("/summarize")
async def summarize(file: UploadFile = File(...)):
    file_bytes = await file.read()

    try:
        summary, marks, saved_file = process_file(file_bytes, file.filename)

        return {
            "filename": file.filename,
            "summary": summary,
            "marks": marks,
            "saved_as": saved_file
        }

    except Exception as e:
        return {"error": str(e)}