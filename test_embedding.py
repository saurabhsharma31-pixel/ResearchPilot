
from google import genai
from google.genai import types
import getpass
import time

api_key = getpass.getpass("Enter Gemini API key: ")

client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(timeout=60000)
)

print("Testing Gemini embeddings...")

start = time.time()

try:
    response = client.models.embed_content(
        model="gemini-embedding-2",
        contents="Summarize my Data Science internship experience.",
        config=types.EmbedContentConfig(
            output_dimensionality=768
        )
    )

    print("Embedding successful!")
    print("Vector dimensions:",
          len(response.embeddings[0].values))
    print("Time taken:",
          round(time.time() - start, 2), "seconds")

except Exception as e:
    print("ERROR:", repr(e))
    print("Time taken:",
          round(time.time() - start, 2), "seconds")
