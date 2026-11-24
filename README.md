# ✦ ResearchPilot — AI-Powered PDF Research Assistant

**Turn your PDFs into an interactive research workspace.**

ResearchPilot is an AI-powered document intelligence application built using Python, Streamlit, and Google's Gemini API. It uses Retrieval-Augmented Generation (RAG) to retrieve relevant passages from uploaded PDF documents and generate answers grounded in the supplied evidence.

🔗 **Live Demo:** https://researchpilot-saurabh.streamlit.app  
📂 **GitHub Repository:** https://github.com/saurabhsharma31-pixel/ResearchPilot

---

## 🚀 Features

### 💬 1. Ask Documents
- Upload and query multiple PDF documents.
- Ask natural-language questions about document content.
- Generate evidence-grounded answers using Gemini.
- View supporting document names, page numbers, and source excerpts.
- Export conversations as Markdown.

### 📚 2. Document Library
- Upload multiple PDF files.
- Extract text page by page using PyMuPDF.
- Create section-aware text chunks.
- Generate vector embeddings for semantic retrieval.
- View indexed documents and searchable chunks.

### ⚖️ 3. Compare PDFs
- Select multiple documents for comparison.
- Compare findings, evidence, and recommendations.
- Generate document-specific insights.
- Inspect the evidence used in comparison answers.

### 📊 4. RAG Evaluation Dashboard
- Test retrieval using custom questions.
- Inspect retrieved passages and their ranking.
- View semantic similarity, keyword relevance, and hybrid ranking scores.
- Label retrieved passages as relevant or irrelevant.
- Calculate human-labelled Precision@K and Mean Reciprocal Rank (MRR).
- Generate answers from retrieved evidence.
- Evaluate answer relevance, faithfulness, completeness, citation support, and overall quality using an AI-assisted evaluator.

---

## 🧠 How It Works

ResearchPilot follows a Retrieval-Augmented Generation pipeline:

1. **PDF Extraction:** PyMuPDF extracts text from uploaded PDFs while retaining page-level metadata.
2. **Intelligent Chunking:** Documents are divided into section-aware text chunks.
3. **Vector Embeddings:** Gemini Embedding 2 converts document passages into 768-dimensional vector representations.
4. **Hybrid Retrieval:** Semantic similarity and keyword matching are combined to rank relevant passages.
5. **Context Diversification:** Near-duplicate passages are filtered and retrieval results are diversified.
6. **Answer Generation:** Gemini 3.1 Flash-Lite generates answers using retrieved document evidence.
7. **Source Attribution:** Supporting document names, page numbers, and excerpts are displayed.
8. **Evaluation:** Retrieval and generated-answer quality can be inspected using the evaluation dashboard.

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Frontend / UI | Streamlit |
| LLM | Google Gemini 3.1 Flash-Lite |
| Embedding Model | Gemini Embedding 2 |
| RAG | Retrieval-Augmented Generation |
| Vector Operations | NumPy |
| PDF Processing | PyMuPDF |
| API Integration | Google GenAI SDK |
| Testing | Pytest |
| Deployment | Streamlit Community Cloud |
| Version Control | Git and GitHub |

---

## ⚙️ Installation and Setup

### Prerequisites
- Python 3.10 or newer
- Google Gemini API key
- Git

### 1. Clone the repository

```bash
git clone https://github.com/saurabhsharma31-pixel/ResearchPilot.git
cd ResearchPilot
```

### 2. Create a virtual environment

**Windows:**

```bash
py -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your Gemini API key

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
```

Alternatively, enter the key in the application sidebar.

Get your API key from [Google AI Studio](https://aistudio.google.com/).

**Security:** Never commit your real `.env` file or expose your API key publicly.

### 5. Run the application

```bash
python -m streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

## 📁 Project Structure

```text
ResearchPilot/
│
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
├── pytest.ini
│
├── .streamlit/
│   └── config.toml
│
├── src/
│   ├── __init__.py
│   ├── pdf_engine.py
│   ├── rag_engine.py
│   ├── utils.py
│   └── test_gemini.py
│
└── tests/
    └── test_pdf_engine.py
```

---

## 📈 RAG Evaluation Methodology

ResearchPilot includes an interactive evaluation workflow.

**Retrieval metrics:**
- Precision@K: Measures the proportion of retrieved passages manually labelled relevant.
- MRR: Measures how highly the first relevant passage is ranked.

**Answer-quality dimensions:**
- Relevance
- Evidence faithfulness
- Completeness
- Citation support
- Overall answer quality

The answer evaluator uses an LLM-as-a-judge approach. Its scores are AI-assisted assessments, not independent ground truth.

---

## ⚠️ Limitations

- The current PDF extraction pipeline supports text-based PDFs; scanned-image OCR is not included.
- Uploaded documents, embeddings, and conversation history are maintained in Streamlit session state rather than a persistent database.
- Refreshing or restarting a session may require uploading and processing documents again.
- Gemini API usage is subject to model availability, quotas, and rate limits.
- AI-generated answers should be verified against their cited source documents.

---

## 🔮 Future Improvements

- Persistent conversation history and searchable chat archives.
- Delete individual recent searches and conversations.
- OCR support for scanned PDF documents.
- Persistent vector storage.
- User authentication and multi-user workspaces.
- Advanced retrieval evaluation using benchmark datasets.
- Improved document parsing and retrieval optimization.

---

## 👨‍💻 Author

**Saurabh Sharma**

B.Tech — Computer Science (Data Science)

Interested in Artificial Intelligence, Machine Learning, Generative AI, and Data Science.

- GitHub: https://github.com/saurabhsharma31-pixel
- LinkedIn: https://www.linkedin.com/in/saurabh-sharma-5a94321ba/

---

⭐ If you find ResearchPilot useful, consider giving this repository a star.

**Built with Python, Gemini, RAG, and Streamlit.**