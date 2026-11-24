from __future__ import annotations

import getpass
import time

from google import genai
from google.genai import types


def main():
    api_key = getpass.getpass("Enter Gemini API key: ").strip()
    if not api_key:
        print("No API key entered.")
        return

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=120_000,
            retry_options=types.HttpRetryOptions(
                attempts=3,
                initial_delay=2.0,
                max_delay=20.0,
                http_status_codes=[408, 429, 500, 502, 503, 504],
            ),
        ),
    )

    print("Testing Gemini Embedding 2...")
    start = time.time()
    try:
        result = client.models.embed_content(
            model="gemini-embedding-2",
            contents="task: question answering | query: What is RAG?",
            config=types.EmbedContentConfig(output_dimensionality=768),
        )
        print(f"Embedding passed: {len(result.embeddings[0].values)} dimensions.")
        print(f"Time: {time.time() - start:.1f}s")
    except Exception as exc:
        print(f"Embedding failed: {type(exc).__name__}: {exc}")
        return

    print("\nTesting Gemini generation...")
    start = time.time()
    try:
        result = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents="Explain RAG in two short sentences.",
            config=types.GenerateContentConfig(temperature=0.2, max_output_tokens=120),
        )
        print(result.text)
        print(f"Time: {time.time() - start:.1f}s")
    except Exception as exc:
        print(f"Generation failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
