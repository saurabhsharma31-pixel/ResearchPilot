from src.pdf_engine import chunk_pages


def test_chunk_pages_preserves_metadata_and_content():
    pages = [{
        "document": "sample.pdf",
        "page": 3,
        "text": "This is a sample document. " * 120,
    }]
    chunks = chunk_pages(pages, chunk_size=180, overlap=30)
    assert len(chunks) > 1
    assert all(c["document"] == "sample.pdf" for c in chunks)
    assert all(c["page"] == 3 for c in chunks)
    assert all(c["text"].strip() for c in chunks)


def test_empty_page_list():
    assert chunk_pages([]) == []
