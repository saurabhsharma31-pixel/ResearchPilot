# ✦ ResearchPilot — AI-Powered PDF Research Assistant

A Retrieval-Augmented Generation (RAG) application that lets you upload multiple PDFs, ask questions, compare documents, and receive grounded answers with document/page citations.

## Features

- Multi-PDF upload and text extraction with PyMuPDF
- Gemini Embedding 2 vector embeddings
- Semantic retrieval using cosine similarity and NumPy
- Grounded answers powered by Gemini 3.8 Flash
- Inline document/page citations and expandable source excerpts
- Cross-document comparison
- Conversation export to Markdown
- Retrieval inspection with cosine similarity scores
- Session-based workspace; no database or infrastructure setup required

## Tech stack

Python · Streamlit · Google Gemini API (`google-genai`) · Gemini Embedding 2 · NumPy · PyMuPDF

## 1. Requirements

- Python 3.10 or newer
- A Google Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)
- Internet connection for Gemini API calls

## 2. Run locally (Windows)

Open the project folder in VS Code.

```powershell
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Open `.env`, replace the placeholder with your key:

```env
GEMINI_API_KEY=your_actual_key
```

Run:

```powershell
python -m streamlit run app.py
```

The app opens at `http://localhost:8501`.

You can alternatively paste the API key into the password field in the sidebar. That key is held only in the current Streamlit session.

## 3. Push to GitHub

**Never upload your real `.env` file or expose your API key.** `.gitignore` already excludes it.

```bash
git init
git add .
git commit -m "Build ResearchPilot AI PDF Research Assistant"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/ResearchPilot.git
git push -u origin main
```

## 4. Deploy free on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create an app and select this repository, branch `main`, and file `app.py`.
4. Open **App settings → Secrets** and add:

```toml
GEMINI_API_KEY = "your_actual_key"
```

5. Deploy. The app reads the key from Streamlit secrets or the `GEMINI_API_KEY` environment variable.

Gemini API usage may be subject to Google's quotas, rate limits, and billing terms. Free availability is not guaranteed for every model or account.

## How it works

1. **Extract:** PyMuPDF extracts selectable text page by page.
2. **Chunk:** Text is divided into overlapping chunks while retaining document and page metadata.
3. **Embed:** Gemini Embedding 2 converts chunks into vector representations.
4. **Retrieve:** A question is embedded and compared with stored vectors using cosine similarity.
5. **Generate:** The most relevant excerpts are passed to Gemini 3.8 Flash with instructions to answer only from supplied evidence.
6. **Cite:** The interface displays source document names and page numbers alongside the answer.

## Important notes

- The app processes text-based PDFs. Scanned image PDFs require OCR; OCR is intentionally not included to keep installation simple.
- Documents and embeddings live in Streamlit session memory. Refreshing/restarting the app or starting a new session may require uploading and processing the PDFs again.
- This starter deployment is intended for personal/demo use. It does not include user accounts, persistent storage, or production-grade multi-user isolation.
- AI answers can be incorrect. Check important statements against the cited PDF pages.
- Keep your API key private and monitor usage in Google AI Studio.

## Project structure

```text
ResearchPilot/
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── src/
│   ├── __init__.py
│   ├── pdf_engine.py
│   ├── rag_engine.py
│   └── utils.py
└── tests/
    └── test_pdf_engine.py
```

## Resume description

**AI-Powered PDF Research Assistant (ResearchPilot)**
- Built a multi-document RAG application using Python, Streamlit, Gemini API, and Gemini Embedding 2 to answer natural-language questions over PDF collections.
- Implemented page-aware PDF extraction, overlapping text chunking, vector embeddings, and cosine-similarity retrieval.
- Integrated Gemini-powered grounded responses with source citations, cross-document comparison, and retrieval-quality inspection.

Only add performance metrics after measuring them on a documented evaluation set.
