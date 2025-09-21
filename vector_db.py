# import pywhatkit
# from datetime import datetime, timedelta

# # WhatsApp number and message
# phone_number = "+17033648283"
# message = "testing_message"

# # Current time + 1 minute (to avoid CallTimeException)
# now = datetime.now() + timedelta(minutes=1)
# hour = now.hour
# minute = now.minute

# # Send WhatsApp message
# pywhatkit.sendwhatmsg(phone_number, message, hour, minute)



# Install required packages first:
# pip install sentence-transformers faiss-cpu numpy

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


with open("file.txt", "r", encoding="utf-8") as f:
    text = f.read()


def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

chunks = split_text(text)
print(f"Total chunks created: {len(chunks)}")


model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = model.encode(chunks, convert_to_numpy=True)
print(f"Embeddings shape: {embeddings.shape}")


dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)  # L2 distance
index.add(embeddings)
print(f"Total vectors in index: {index.ntotal}")

# Save the index and chunks for future use
faiss.write_index(index, "vector_index.faiss")
np.save("chunks.npy", chunks)
print("Vector database saved successfully!")

def query_vector_db(query, top_k=5):
    # Load index and chunks if needed
    index = faiss.read_index("vector_index.faiss")
    chunks = np.load("chunks.npy", allow_pickle=True)

   
    query_vec = model.encode([query], convert_to_numpy=True)


    D, I = index.search(query_vec, k=top_k)
    results = [chunks[i] for i in I[0]]
    return results


if __name__ == "__main__":
    user_query = input("Enter your query: ")
    results = query_vector_db(user_query, top_k=5)
    print("\nTop matching chunks:\n")
    for r in results:
        print(r)
        print("-"*50)


