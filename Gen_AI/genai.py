import os
import pandas as pd
import docx
from pathlib import Path
import google.generativeai as genai
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
import time

# Configure Gemini API
genai.configure(api_key="AIzaSyAkdFOq1j6Gl2bbbwMhzldHq5VkxCAyE1Q")
model = genai.GenerativeModel("gemini-1.5-pro")

# Path to documents
DOCS_FOLDER = r"C:\Users\Meghananda\Downloads\Finacle user manual"

# Load documents
def load_documents(folder_path):
    docs = []
    for file in Path(folder_path).rglob("*"):
        if file.suffix.lower() == ".txt":
            with open(file, "r", encoding="utf-8") as f:
                docs.append((str(file), f.read()))
        elif file.suffix.lower() == ".docx":
            doc = docx.Document(file)
            full_text = "\n".join([p.text for p in doc.paragraphs])
            docs.append((str(file), full_text))
        elif file.suffix.lower() in [".xls", ".xlsx"]:
            try:
                df = pd.read_excel(file)
                docs.append((str(file), df.to_string()))
            except Exception as e:
                print(f"Failed to read {file}: {e}")
    return docs

# Split into chunks
def split_into_chunks(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Estimate token count
def estimate_tokens(text):
    return int(len(text.split()) / 0.75)

# Truncate to token limit
def truncate_context_to_token_limit(context_chunks, token_limit):
    combined = ""
    total_tokens = 0
    for chunk in context_chunks:
        chunk_tokens = estimate_tokens(chunk)
        if total_tokens + chunk_tokens > token_limit:
            break
        combined += chunk + "\n\n"
        total_tokens += chunk_tokens
    return combined, total_tokens

# Build vector store
def build_vector_store(documents, db_path="rag_db"):
    client = PersistentClient(path=db_path)
    collection = client.get_or_create_collection(
        name="finacle_docs",
        embedding_function=embedding_functions.GoogleGenerativeAiEmbeddingFunction(
            api_key="AIzaSyAkdFOq1j6Gl2bbbwMhzldHq5VkxCAyE1Q",
            model_name="models/embedding-001"
        )
    )

    for idx, (filename, content) in enumerate(documents):
        chunks = split_into_chunks(content)
        for j, chunk in enumerate(chunks):
            collection.add(
                documents=[chunk],
                metadatas=[{"filename": filename}],
                ids=[f"{idx}_{j}"]
            )
    return collection

# Generate test scenario
def generate_test_scenario(user_input, collection, top_k=3, token_limit=1000):
    results = collection.query(query_texts=[user_input], n_results=top_k)
    raw_chunks = [doc for doc in results["documents"][0]]

    context, used_tokens = truncate_context_to_token_limit(raw_chunks, token_limit)
    print(f"[INFO] Approx. tokens in prompt context: {used_tokens} / {token_limit}")

    prompt = f"""You are a Finacle test engineer.
Using the following documentation chunks:\n\n{context}\n\n
Generate detailed test scenarios based on this user request: "{user_input}"
"""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"[ERROR] {e}"

# Main program
if __name__ == "__main__":
    print("🔍 Loading and processing Finacle documents...")
    docs = load_documents(DOCS_FOLDER)
    collection = build_vector_store(docs)

    try:
        user_token_limit = int(input("Enter maximum prompt token limit (e.g., 800, 1000): "))
    except:
        user_token_limit = 1000

    print("✅ Ready to generate Finacle test scenarios!")
    while True:
        query = input("\nEnter your Finacle test scenario request (or type 'exit'): ")
        if query.lower() == "exit":
            break
        scenario = generate_test_scenario(query, collection, top_k=3, token_limit=user_token_limit)
        print("\n📄 Generated Test Scenario:\n")
        print(scenario)
        time.sleep(5)  # avoid rapid-fire requests
