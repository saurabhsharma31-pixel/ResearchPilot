import os
import re
import time
import streamlit as st
from dotenv import load_dotenv

from src.pdf_engine import extract_pdf_pages, chunk_pages
from src.rag_engine import GeminiRAG
from src.utils import build_markdown_export

load_dotenv()

st.set_page_config(
    page_title="ResearchPilot | PDF Research Assistant",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

/* App canvas and typography */
html, body, [class*="css"], .stApp {
    font-family: 'DM Sans', sans-serif !important;
}
.stApp {
    background: #0b1020 !important;
    color: #edf2ff !important;
}
.main .block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Force readable foreground colors across Streamlit's generated elements */
.stApp, .stApp p, .stApp span, .stApp label,
.stApp li, .stApp small, .stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stMarkdownContainer"] li {
    color: #e8eefc !important;
}
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
    color: #f7f9ff !important;
    font-family: 'Manrope', sans-serif !important;
    letter-spacing: -0.025em;
}
.stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stCaptionContainer"] * {
    color: #a9b8d3 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #111a2e !important;
    border-right: 1px solid #263653 !important;
}
[data-testid="stSidebar"] * {
    color: #e8eefc !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #c4d0e6 !important;
}

/* Hero */
.hero {
    padding: 1.8rem 2rem;
    border: 1px solid #34496e;
    border-radius: 22px;
    background: linear-gradient(120deg, #172b4a 0%, #1b2544 58%, #30234b 100%);
    margin-bottom: 1.5rem;
    box-shadow: 0 16px 45px rgba(0,0,0,.18);
}
.hero h1 {
    font-family: 'Manrope', sans-serif;
    font-size: clamp(2rem, 4vw, 2.8rem);
    font-weight: 800;
    margin: .25rem 0 .45rem;
    color: #ffffff !important;
}
.hero p { color: #c6d4ee !important; margin: 0; font-size: 1.02rem; }
.eyebrow {
    color: #a9b9ff !important;
    text-transform: uppercase;
    letter-spacing: .16em;
    font-size: .72rem;
    font-weight: 700;
}

/* Cards and metrics */
.metric, .source {
    background: #141f35;
    border: 1px solid #2b3d5d;
    border-radius: 15px;
    padding: 15px 17px;
    color: #e8eefc !important;
}
.metric .num {
    font-family: 'Manrope', sans-serif;
    font-size: 1.7rem;
    font-weight: 800;
    color: #ffffff !important;
}
.metric .lbl, .source small { color: #aebdd7 !important; }
.source { margin: .55rem 0; line-height: 1.55; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid #344461;
}
.stTabs [data-baseweb="tab"] {
    color: #c6d2e8 !important;
    background: transparent !important;
    border-radius: 10px 10px 0 0;
    padding: 12px 16px;
}
.stTabs [data-baseweb="tab"] p,
.stTabs [data-baseweb="tab"] div {
    color: #c6d2e8 !important;
}
.stTabs [aria-selected="true"] {
    background: #1a2944 !important;
}
.stTabs [aria-selected="true"] p,
.stTabs [aria-selected="true"] div {
    color: #ffffff !important;
}

/* Inputs */
.stTextInput label, .stTextArea label, .stFileUploader label,
.stSelectbox label, .stMultiSelect label {
    color: #dce6f8 !important;
}
.stTextInput input, .stTextArea textarea, [data-baseweb="select"] > div {
    background: #f7f9fd !important;
    color: #172033 !important;
    border: 1px solid #bdc9dc !important;
    border-radius: 10px !important;
}
.stTextInput input::placeholder, .stTextArea textarea::placeholder {
    color: #65738b !important;
}
[data-testid="stChatInput"] textarea {
    color: #172033 !important;
}
[data-testid="stChatInput"] {
    border-color: #60769d !important;
}
.stButton button, .stDownloadButton button {
    border-radius: 10px !important;
    font-weight: 700 !important;
    min-height: 2.65rem;
}
.stButton button[kind="primary"] {
    background: #6179ee !important;
    border: 1px solid #8194ff !important;
    color: #ffffff !important;
}
.stButton button[kind="secondary"], .stDownloadButton button {
    background: #1b2a45 !important;
    border: 1px solid #3a5278 !important;
    color: #f2f6ff !important;
}

/* Chat messages, alerts, expanders */
div[data-testid="stChatMessage"] {
    background: #141f35 !important;
    border: 1px solid #2b3d5d !important;
    border-radius: 15px !important;
}
div[data-testid="stAlert"] {
    border-radius: 12px !important;
}
.streamlit-expanderHeader {
    color: #eaf0ff !important;
    background: #17243b !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploader"] {
    background: #121e33;
    border: 1px dashed #52688d;
    border-radius: 14px;
    padding: 12px;
}
hr { border-color: #2d3e5d !important; }

/* Keep general app text light, but never apply this to form controls. */
.stApp p, .stApp span, .stApp label, .stApp li,
.stApp [data-testid="stMarkdownContainer"] {
    color: #e8eefc;
}

/* Sidebar password field: light field with dark, readable masked/revealed text. */
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] input[type="password"],
[data-testid="stSidebar"] input[type="text"] {
    background: #f7f9fd !important;
    color: #172033 !important;
    -webkit-text-fill-color: #172033 !important;
    caret-color: #172033 !important;
    opacity: 1 !important;
}

/* Main chat composer: dark field with clearly visible white typed text. */
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] input,
[data-testid="stChatInput"] [contenteditable="true"],
[data-testid="stChatInput"] [role="textbox"] {
    background: #162238 !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    caret-color: #ffffff !important;
    opacity: 1 !important;
    visibility: visible !important;
    text-shadow: none !important;
}

[data-testid="stChatInput"] textarea::placeholder,
[data-testid="stChatInput"] input::placeholder {
    color: #94a3b8 !important;
    -webkit-text-fill-color: #94a3b8 !important;
    opacity: 1 !important;
}

/* Other regular text inputs and text areas use dark text on light backgrounds. */
.stTextInput input,
.stTextArea textarea,
[data-baseweb="select"] input {
    background: #f7f9fd !important;
    color: #172033 !important;
    -webkit-text-fill-color: #172033 !important;
    caret-color: #172033 !important;
    opacity: 1 !important;
}
.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
    color: #65738b !important;
    -webkit-text-fill-color: #65738b !important;
}

/* Tabs and buttons retain the light-on-dark theme. */
.stTabs button[data-baseweb="tab"],
.stTabs button[data-baseweb="tab"] *,
.stApp button,
.stApp button * {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

if "documents" not in st.session_state:
    st.session_state.documents = []
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "rag" not in st.session_state:
    st.session_state.rag = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_eval_results" not in st.session_state:
    st.session_state.rag_eval_results = None
if "rag_eval_labels" not in st.session_state:
    st.session_state.rag_eval_labels = {}
if "api_key" not in st.session_state:
    secret_key = ""
    try:
        secret_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        pass
    st.session_state.api_key = os.getenv("GEMINI_API_KEY", "") or secret_key

with st.sidebar:
    # Inject sidebar-specific styles from inside the sidebar container.
    st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        --text-color: #ffffff !important;
        --secondary-text-color: #e3ebfa !important;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] * {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {
        color: #e3ebfa !important;
        -webkit-text-fill-color: #e3ebfa !important;
    }
    [data-testid="stSidebar"] input {
        color: #172033 !important;
        -webkit-text-fill-color: #172033 !important;
    }
    [data-testid="stSidebar"] button,
    [data-testid="stSidebar"] button * {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown("## ✦ ResearchPilot")
    st.caption("Your AI workspace for understanding documents")
    st.markdown("---")
    st.markdown("### Google Gemini API")
    key = st.text_input(
        "Gemini API key",
        value=st.session_state.api_key,
        type="password",
        help="Read from GEMINI_API_KEY in .env or enter it here for this session.",
        placeholder="Paste your Google AI Studio API key",
    )
    if key != st.session_state.api_key:
        st.session_state.api_key = key
        st.session_state.rag = None
    st.caption("Your key stays in this session and is not stored in the project.")
    st.markdown("---")
    st.markdown("### Your library")
    st.write(f"**{len(st.session_state.documents)}** document(s) loaded")
    if st.session_state.documents:
        for doc in st.session_state.documents:
            st.caption(f"📄 {doc['name']} · {doc['pages']} pages")
    if st.button("Clear workspace", use_container_width=True):
        st.session_state.documents = []
        st.session_state.chunks = []
        st.session_state.rag = None
        st.session_state.messages = []
        st.session_state.rag_eval_results = None
        st.session_state.rag_eval_labels = {}
        st.session_state.pop("chat_document_scope", None)
        st.session_state.pop("rag_eval_document_scope", None)
        st.rerun()
    st.markdown("---")
    st.caption("Built with Gemini · RAG · Streamlit")

st.markdown("""
<div class="hero">
  <div class="eyebrow">AI DOCUMENT INTELLIGENCE</div>
  <h1>ResearchPilot</h1>
  <p>Upload your PDFs. Ask better questions. Get grounded answers with page-level sources.</p>
</div>
""", unsafe_allow_html=True)

tab_chat, tab_library, tab_compare, tab_eval = st.tabs(
    ["💬 Ask documents", "📚 Document library", "⚖️ Compare PDFs", "📊 RAG evaluation"]
)

def get_engine():
    if not st.session_state.api_key.strip():
        st.error("Add your Google Gemini API key in the sidebar to process documents and ask questions.")
        return None
    if st.session_state.rag is None:
        try:
            st.session_state.rag = GeminiRAG(st.session_state.api_key.strip())
        except Exception as exc:
            st.error(f"Could not initialize Gemini: {exc}")
            return None
    return st.session_state.rag

def process_uploads(uploaded_files):
    engine = get_engine()
    if engine is None:
        return
    existing = {d["name"] for d in st.session_state.documents}
    new_files = [f for f in uploaded_files if f.name not in existing]
    if not new_files:
        st.info("Those files are already in your current library.")
        return
    all_new_chunks, new_docs = [], []
    progress = st.progress(0, text="Reading your PDFs...")
    try:
        for idx, uploaded in enumerate(new_files):
            pages = extract_pdf_pages(uploaded.getvalue(), uploaded.name)
            if not pages:
                st.warning(f"No extractable text found in {uploaded.name}. It may be scanned; OCR is not included.")
                continue
            chunks = chunk_pages(pages)
            all_new_chunks.extend(chunks)
            new_docs.append({"name": uploaded.name, "pages": len(pages), "chunks": len(chunks)})
            progress.progress((idx + 1) / len(new_files), text=f"Prepared {uploaded.name}")
        if not all_new_chunks:
            progress.empty()
            return
        with st.spinner("Creating Google Gemini embeddings..."):
            vectors = engine.embed_texts(
                [c["text"] for c in all_new_chunks],
                input_type="document",
                titles=[f"{c.get('document', 'none')} | {c.get('section', 'General')}" for c in all_new_chunks],
            )

        # Never silently index chunks without vectors: zip() truncates if the
        # embedding API returns fewer vectors than requested.
        if len(vectors) != len(all_new_chunks):
            raise RuntimeError(
                f"Gemini returned {len(vectors)} embeddings for "
                f"{len(all_new_chunks)} PDF chunks. No chunks were added. "
                "Please retry processing the PDF."
            )
        for chunk, vector in zip(all_new_chunks, vectors):
            if vector is None or len(vector) == 0:
                raise RuntimeError(
                    "Gemini returned an empty embedding for a PDF chunk. "
                    "Please retry processing the PDF."
                )
            chunk["embedding"] = vector
            chunk["embedding_version"] = engine.EMBEDDING_VERSION

        # Defensive check before making the new chunks visible to the app.
        if any("embedding" not in chunk for chunk in all_new_chunks):
            raise RuntimeError("At least one PDF chunk is missing its embedding.")
        st.session_state.chunks.extend(all_new_chunks)
        st.session_state.documents.extend(new_docs)
        progress.empty()
        st.success(f"Added {len(new_docs)} document(s) and indexed {len(all_new_chunks)} text chunks.")

        # The sidebar is rendered before the Document Library tab.
        # Rerun once processing succeeds so "Your library" refreshes immediately.
        st.rerun()
    except Exception as exc:
        
        
         import traceback

         st.error(f"Gemini request failed: {type(exc).__name__}: {exc}")

         with st.expander("Technical error details", expanded=True):
             
             st.code(traceback.format_exc())

def clean_inline_citations(text):
    """Remove repetitive inline PDF/page markers; detailed evidence stays in View sources."""
    if not text:
        return text
    # Handles markers such as [Resume.pdf, p. 1] and [Source 2, p. 3; Source 4, p. 1].
    text = re.sub(
        r"\s*\[(?=[^\]]*(?:\.pdf|Source\s*\d+))(?=[^\]]*(?:p\.\s*\d+|page\s*\d+))[^\]]*\]",
        "",
        text,
        flags=re.IGNORECASE,
    )
    # Tidy whitespace left where a citation was removed.
    text = re.sub(r"[ \t]+([,.;:])", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunks_for_scope(scope: str) -> list[dict]:
    """Return indexed chunks for the chosen document scope."""
    if scope == "All documents":
        return st.session_state.chunks
    return [
        chunk for chunk in st.session_state.chunks
        if chunk.get("document") == scope
    ]


with tab_chat:
    left, right = st.columns([1.55, 1], gap="large")
    with left:
        st.markdown("### Ask your research question")

        document_options = ["All documents"] + [
            doc["name"] for doc in st.session_state.documents
        ]
        chat_scope = st.selectbox(
            "Search in",
            document_options,
            help="Choose whether to search every uploaded PDF or restrict answers to one document.",
            key="chat_document_scope",
            disabled=not bool(st.session_state.documents),
        )
        scoped_chat_chunks = chunks_for_scope(chat_scope)

        # Keep the composer above the conversation so it does not move below
        # older answers as the chat history grows.
        question = st.chat_input(
            "Ask something about your PDFs...",
            disabled=not bool(st.session_state.chunks),
            key="researchpilot_chat_input",
        )

        # Rebuild question/answer pairs and show the newest question first.
        # Each question is a collapsible container: click to open/close its answer.
        conversation_pairs = []
        message_index = 0
        while message_index < len(st.session_state.messages):
            current = st.session_state.messages[message_index]
            if current.get("role") == "user":
                answer_message = None
                if (
                    message_index + 1 < len(st.session_state.messages)
                    and st.session_state.messages[message_index + 1].get("role") == "assistant"
                ):
                    answer_message = st.session_state.messages[message_index + 1]
                    message_index += 1
                conversation_pairs.append((current.get("content", "Question"), answer_message))
            message_index += 1

        for saved_question, saved_answer in reversed(conversation_pairs):
            with st.expander(f"💬 {saved_question}", expanded=False):
                if saved_answer:
                    st.markdown(clean_inline_citations(saved_answer.get("content", "")))
                    if saved_answer.get("sources"):
                        with st.expander("View sources", expanded=False):
                            for source in saved_answer["sources"]:
                                st.markdown(
                                    f"**{source['document']} · Page {source['page']} · {source.get('section', 'General')}**  \n{source['excerpt']}"
                                )
                else:
                    st.info("No answer was saved for this question.")

        if question:
            engine = get_engine()
            if engine:
                with st.spinner("Searching your documents and composing an answer..."):
                    try:
                        answer, sources = engine.answer(question, scoped_chat_chunks)
                        clean_answer = clean_inline_citations(answer)
                        st.session_state.messages.extend([
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": clean_answer, "sources": sources},
                        ])
                        # Refresh the chat so the new Q&A appears at the top.
                        st.rerun()
                    except Exception as exc:
                        st.session_state.messages.extend([
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": f"Gemini request failed: {exc}"},
                        ])
                        st.rerun()

    with right:
        st.markdown("### Start here")
        st.markdown("""
        - **Summarize:** What are the main ideas in these documents?
        - **Find evidence:** What do the documents say about a specific claim?
        - **Understand:** Explain a difficult concept in simple terms.
        - **Compare:** What differs across the uploaded reports?
        """)
        if not st.session_state.chunks:
            st.info("Upload PDFs in the Document library tab to begin.")
        else:
            st.markdown("### Workspace snapshot")
            c1, c2 = st.columns(2)
            c1.markdown(f'<div class="metric"><div class="num">{len(st.session_state.documents)}</div><div class="lbl">Documents</div></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="metric"><div class="num">{len(st.session_state.chunks)}</div><div class="lbl">Indexed chunks</div></div>', unsafe_allow_html=True)
            if st.session_state.messages:
                export = build_markdown_export(st.session_state.messages)
                st.download_button("⬇ Export conversation", export, "researchpilot_conversation.md", "text/markdown", use_container_width=True)

with tab_library:
    st.markdown("### Build your document library")
    st.write("Upload one or more text-based PDF files. Documents are processed in this browser session.")
    uploads = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="For best results, use PDFs with selectable text. Scanned-image OCR is not included.",
    )
    if st.button("Process PDFs", type="primary", disabled=not uploads):
        process_uploads(uploads)
    if st.session_state.documents:
        st.markdown("#### Indexed documents")
        for doc in st.session_state.documents:
            st.markdown(
                f'<div class="source">📄 <b>{doc["name"]}</b><br><small>{doc["pages"]} pages · {doc["chunks"]} searchable chunks</small></div>',
                unsafe_allow_html=True,
            )
    else:
        st.info("Your library is empty. Upload PDFs above.")

with tab_compare:
    st.markdown("### Compare information across documents")
    if len(st.session_state.documents) < 2:
        st.info("Upload at least two PDFs to use document comparison.")
    else:
        names = [d["name"] for d in st.session_state.documents]
        selected = st.multiselect("Choose documents to compare", names, default=names[:2])
        prompt = st.text_area("What would you like to compare?", "Compare the main findings, evidence, and recommendations.")
        if st.button("Compare selected PDFs", type="primary", disabled=len(selected) < 2 or not prompt.strip()):
            engine = get_engine()
            if engine:
                with st.spinner("Retrieving evidence from the selected PDFs..."):
                    try:
                        subset = [c for c in st.session_state.chunks if c["document"] in selected]
                        answer, sources = engine.answer(prompt, subset, compare=True)
                        st.markdown(clean_inline_citations(answer))
                        with st.expander("Evidence used", expanded=False):
                            for s in sources:
                                st.markdown(f"**{s['document']} · Page {s['page']} · {s.get('section', 'General')}**  \n{s['excerpt']}")
                    except Exception as exc:
                        st.error(f"Comparison failed: {exc}")

with tab_eval:
    st.markdown("### RAG retrieval diagnostics")
    st.write(
        "Test how well ResearchPilot finds passages in your uploaded PDFs. "
        "Similarity scores explain ranking; they are not accuracy or faithfulness scores."
    )

    eval_document_options = ["All documents"] + [
        doc["name"] for doc in st.session_state.documents
    ]
    eval_scope = st.selectbox(
        "Evaluate retrieval in",
        eval_document_options,
        help="Restrict this test to one PDF or search the complete library.",
        key="rag_eval_document_scope",
        disabled=not bool(st.session_state.documents),
    )
    scoped_eval_chunks = chunks_for_scope(eval_scope)

    eval_q = st.text_input(
        "Test question",
        placeholder="Example: What internships have I completed?",
        key="rag_eval_question",
    )
    eval_top_k = st.slider(
        "Passages to retrieve", min_value=3, max_value=10, value=5, key="rag_eval_top_k"
    )

    if st.button(
        "Run retrieval test",
        type="primary",
        disabled=not eval_q.strip() or not scoped_eval_chunks,
        key="run_rag_diagnostics",
    ):
        engine = get_engine()
        if engine:
            with st.spinner("Embedding the question and ranking relevant passages..."):
                try:
                    started = time.perf_counter()
                    results = engine.retrieve(eval_q, scoped_eval_chunks, top_k=eval_top_k)
                    elapsed_ms = (time.perf_counter() - started) * 1000
                    st.session_state.rag_eval_results = {
                        "question": eval_q.strip(),
                        "scope": eval_scope,
                        "results": results,
                        "elapsed_ms": elapsed_ms,
                    }
                    st.session_state.rag_eval_labels = {}
                except Exception as exc:
                    st.error(f"Retrieval test failed: {type(exc).__name__}: {exc}")

    evaluation = st.session_state.get("rag_eval_results")
    if evaluation:
        results = evaluation["results"]
        unique_pages = len({(r["document"], r["page"]) for r in results})
        unique_docs = len({r["document"] for r in results})
        best_score = max(
            (r.get("semantic_score", r.get("score", 0.0)) for r in results),
            default=0.0,
        )

        st.markdown("#### Retrieval overview")
        st.caption(f"Search scope: **{evaluation.get('scope', 'All documents')}**")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Passages found", len(results))
        m2.metric("Unique pages", unique_pages)
        m3.metric("Documents represented", unique_docs)
        m4.metric("Retrieval time", f"{evaluation['elapsed_ms']:.0f} ms")

        if results:
            st.caption(
                f"Best semantic cosine score: {best_score:.3f}. "
                "Mark passages relevant below to calculate human-labelled Precision@K and MRR."
            )
            st.markdown("#### Retrieved passages")
            labels = st.session_state.setdefault("rag_eval_labels", {})
            for rank, result in enumerate(results, start=1):
                passage_key = f"{hash(evaluation['question'])}:{rank}:{result['document']}:{result['page']}:{hash(result['text'])}"
                with st.container(border=True):
                    st.markdown(f"**#{rank} · {result['document']} · Page {result['page']} · {result.get('section', 'General')}**")
                    a, b, c = st.columns(3)
                    a.metric("Semantic", f"{result.get('semantic_score', result.get('score', 0.0)):.3f}")
                    b.metric("Keyword", f"{result.get('keyword_score', 0.0):.3f}")
                    c.metric("Hybrid rank", f"{result.get('hybrid_score', 0.0):.3f}")
                    st.write(result["text"])
                    labels[passage_key] = st.checkbox(
                        "This passage is relevant to my test question",
                        value=labels.get(passage_key, False),
                        key=f"rag_relevant_{passage_key}",
                    )

            relevant_ranks = [
                rank for rank, result in enumerate(results, start=1)
                if labels.get(
                    f"{hash(evaluation['question'])}:{rank}:{result['document']}:{result['page']}:{hash(result['text'])}",
                    False,
                )
            ]
            precision_at_k = len(relevant_ranks) / len(results) if results else 0.0
            reciprocal_rank = 1 / min(relevant_ranks) if relevant_ranks else 0.0

            st.markdown("#### Human-labelled retrieval metrics")
            p1, p2 = st.columns(2)
            p1.metric(f"Precision@{len(results)}", f"{precision_at_k:.2f}")
            p2.metric("MRR", f"{reciprocal_rank:.2f}")
            st.caption(
                "These metrics use your relevance labels, not automatic ground truth. "
                "Recall needs a complete list of all relevant passages. Answer faithfulness "
                "needs a separate answer-evaluation step."
            )
        else:
            st.warning("No passages were retrieved. Try a more specific question or check that your PDFs were processed.")
    elif not scoped_eval_chunks:
        st.info("Upload and process at least one PDF in Document Library before running a retrieval test.")
    else:
        st.info("Enter a question and run a retrieval test to inspect the passages selected by RAG.")

st.markdown("---")
st.caption("ResearchPilot is an AI research aid. Verify important claims against the cited source documents.")
