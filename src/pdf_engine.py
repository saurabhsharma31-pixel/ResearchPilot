import io
import re
import fitz


BULLET_RE = re.compile(r"^\s*(?:[•●▪◦‣]|[-*]\s+|\d+[.)]\s+)")
HEADING_RE = re.compile(
    r"^\s*(?:(?:\d+(?:\.\d+)*\.?\s+)?"
    r"(?:ABSTRACT|INTRODUCTION|BACKGROUND|RELATED WORK|METHODOLOGY|METHODS|"
    r"EXPERIMENTS?|RESULTS|DISCUSSION|CONCLUSION|CONCLUSIONS|REFERENCES|"
    r"SUMMARY|SKILLS|EXPERIENCE|EDUCATION|CERTIFICATIONS|PROJECTS|"
    r"TECHNICAL SKILLS|LANGUAGES\s*&\s*DATA|MACHINE LEARNING|LLMS?\s*&\s*AI|"
    r"VISUALIZATION\s*&\s*BI))\s*:?\s*$",
    re.IGNORECASE,
)


def extract_pdf_pages(pdf_bytes: bytes, filename: str) -> list[dict]:
    """Extract selectable text page-by-page and normalize common PDF artifacts."""
    pages = []
    with fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf") as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text") or ""
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
            text = re.sub(r"[\t\u00a0 ]+", " ", text)
            text = re.sub(r" *\n *", "\n", text)
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            if text:
                pages.append({
                    "document": filename,
                    "page": page_number,
                    "text": text,
                })
    return pages


def _is_heading(line: str) -> bool:
    line = line.strip()
    if not line:
        return False
    if HEADING_RE.match(line):
        return True
    # Academic papers often use numbered headings such as "3.2 Attention".
    if re.match(r"^\d+(?:\.\d+)*\s+[A-Z][A-Za-z0-9 ,:&()/-]{1,65}$", line):
        return True
    return False


def _logical_blocks(text: str) -> list[tuple[str, bool]]:
    """Return (text, is_heading) blocks, keeping wrapped lines and bullets intact."""
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    # Some academic PDFs put a subsection number on its own line.
    merged_lines = []
    i = 0
    while i < len(lines):
        if (
            re.fullmatch(r"\d+(?:\.\d+)+", lines[i])
            and i + 1 < len(lines)
            and 2 <= len(lines[i + 1]) <= 70
        ):
            merged_lines.append(f"{lines[i]} {lines[i + 1]}")
            i += 2
        else:
            merged_lines.append(lines[i])
            i += 1
    lines = merged_lines
    blocks: list[tuple[str, bool]] = []
    current = ""

    def flush():
        nonlocal current
        if current.strip():
            blocks.append((current.strip(), False))
        current = ""

    for line in lines:
        if not line:
            flush()
            continue
        # Suppress standalone PDF page-number lines, which otherwise become
        # meaningless one-character chunks near section boundaries.
        if re.fullmatch(r"\d{1,3}", line):
            continue

        if _is_heading(line):
            flush()
            blocks.append((line, True))
        elif BULLET_RE.match(line):
            flush()
            current = line
        elif current:
            current = f"{current} {line}"
        else:
            current = line

    flush()
    return blocks


def _split_oversized_block(block: str, max_chars: int) -> list[str]:
    """Split an oversized paragraph at sentence or word boundaries, never mid-word."""
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9•●▪◦])", block)
    pieces: list[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            words = sentence.split()
            part = ""
            for word in words:
                candidate = f"{part} {word}".strip()
                if len(candidate) > max_chars and part:
                    pieces.append(part)
                    part = word
                else:
                    part = candidate
            if part:
                current = part
        else:
            candidate = f"{current} {sentence}".strip() if current else sentence
            if len(candidate) > max_chars and current:
                pieces.append(current)
                current = sentence
            else:
                current = candidate

    if current:
        pieces.append(current)
    return pieces


def chunk_pages(
    pages: list[dict],
    chunk_size: int = 1000,
    overlap: int = 0,
) -> list[dict]:
    """Create section-aware chunks with headings attached to their own content.

    The overlap argument is retained for compatibility. It is intentionally not
    used because character overlap can create clipped sentence fragments.
    Returned chunks keep the document/page/text keys used by ResearchPilot and
    add a section field for more informative retrieval and source display.
    """
    if chunk_size < 200:
        raise ValueError("chunk_size must be at least 200 characters.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size.")

    chunks: list[dict] = []
    for page in pages:
        page_text = (page.get("text") or "").strip()
        if not page_text:
            continue

        blocks = _logical_blocks(page_text)
        section = "General"
        current = ""
        page_chunks: list[tuple[str, str]] = []

        def flush():
            nonlocal current
            if current.strip():
                page_chunks.append((section, current.strip()))
            current = ""

        for block, is_heading in blocks:
            if is_heading:
                flush()
                section = block
                # Keep the heading with the first content block in this section.
                current = block
                continue

            prefix = section if section != "General" and len(section) < 100 else ""
            unit_limit = max(200, chunk_size - len(prefix) - 1) if prefix else chunk_size
            units = [block] if len(block) <= unit_limit else _split_oversized_block(block, unit_limit)
            for unit in units:
                candidate = f"{current}\n{unit}".strip() if current else unit
                if len(candidate) <= chunk_size:
                    current = candidate
                else:
                    flush()
                    # Repeat the section title as context when a section spans
                    # multiple chunks, but do not repeat it if it is very long.
                    prefix = section if section != "General" and len(section) < 100 else ""
                    current = f"{prefix}\n{unit}".strip() if prefix else unit

        flush()

        for section_name, piece in page_chunks:
            chunks.append({
                "document": page["document"],
                "page": page["page"],
                "section": section_name,
                "text": piece,
            })

    return chunks
