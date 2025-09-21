
import os
import json
from sentence_transformers import SentenceTransformer
import chromadb


def main():
    folder = r"/Users/yousaf.z/Desktop/self_app_congressional/self_app_congressional/file.txt"
    jsonl_files = [f for f in os.listdir(folder) if f.endswith(".txt")]

    data = []

    # 1. Load all JSONL files
    for file in jsonl_files:
        file_path = os.path.join(folder, file)
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line.strip()))
        print(f"📄 Loaded {file_path}")

    print(f"📄 Total records loaded: {len(data)}")

    # 2. Load biomedical embedding model
    model = SentenceTransformer("lokeshch19/ModernPubMedBERT")

    # 3. Extract abstracts for embedding
    texts = [d["abstract"] for d in data]

    # 4. Generate embeddings
    embeddings = model.encode(texts, show_progress_bar=True)

    # 5. Initialize persistent ChromaDB client (use main DB path)
    db_path = r"/Users/yousaf.z/Desktop/self_app_congressional/self_app_congressional/file.txt"
    os.makedirs(db_path, exist_ok=True)
    client = chromadb.PersistentClient(path=db_path)

    # 6. Create or get the collection
    collection = client.get_or_create_collection("medical_abstracts")

    # 7. Add documents + embeddings + metadata to ChromaDB in chunks
    MAX_BATCH = 5000  # below Chroma's max (5461)
    total = len(data)
    for start in range(0, total, MAX_BATCH):
        end = min(start + MAX_BATCH, total)
        batch_ids = [f"doc_{i}" for i in range(start, end)]
        batch_docs = texts[start:end]
        batch_embs = embeddings[start:end].tolist()
        batch_metas = [{
            "title": d.get("title", ""),
            "abstract": d.get("abstract", ""),
            "pmid": d.get("pmid", "")
        } for d in data[start:end]]

        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            embeddings=batch_embs,
            metadatas=batch_metas
        )
        print(f"✅ Inserted {end - start} records ({end}/{total})")

    print(f"🎉 Completed inserting {total} documents into ChromaDB at {db_path}")


if __name__ == "__main__":
    main()
