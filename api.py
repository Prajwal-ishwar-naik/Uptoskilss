import os
import re
import io
import spacy
import PyPDF2
import docx

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from transformers import pipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------- LOAD MODELS ----------------
nlp = spacy.load("en_core_web_sm")

summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    device=-1
)

# ---------------- APP INIT ----------------
app = FastAPI()
templates = Jinja2Templates(directory="templates")

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

# ---------------- PROCESSING ----------------
def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def sentence_split(text):
    doc = nlp(text)
    return [s.text.strip() for s in doc.sents if len(s.text.strip()) > 5]

def chunk_text(sentences, chunk_size=600):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=100
    )
    return splitter.split_text(" ".join(sentences))

def summarize_chunks(chunks):
    summaries = []

    for chunk in chunks:
        if len(chunk.split()) < 50:
            continue

        summary = summarizer(
            chunk,
            max_length=150,
            min_length=50,
            do_sample=False
        )[0]["summary_text"]

        summaries.append(summary)

    if not summaries:
        return "Text too short to summarize."

    return " ".join(summaries)

def save_summary(filename, summary):
    os.makedirs("outputs", exist_ok=True)
    safe_name = filename.replace(" ", "_")
    path = os.path.join("outputs", f"result_{safe_name}.txt")

    with open(path, "w", encoding="utf-8") as f:
        f.write("===== SUMMARY =====\n\n")
        f.write(summary)

    return path

# ---------------- ROUTES ----------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/summarize", response_class=HTMLResponse)
async def summarize(request: Request, file: UploadFile = File(...)):

    file_bytes = await file.read()

    raw_text = extract_text(file_bytes, file.filename)
    clean = clean_text(raw_text)
    sentences = sentence_split(clean)
    chunks = chunk_text(sentences)
    summary = summarize_chunks(chunks)
    saved_file = save_summary(file.filename, summary)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "summary": summary,
            "saved_file": saved_file
        }
    )
