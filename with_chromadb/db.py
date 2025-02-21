# db.py
import chromadb
import uuid
import numpy as np
from sentence_transformers import SentenceTransformer

# Initialize ChromaDB client
client = chromadb.HttpClient(host="localhost", port=8000)

# Create or get the collection
collection = client.get_or_create_collection("allminilm-test-1")

# Load Nomic embedding model
nomic_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)

def add_to_collection(text_chunks, filename):
    """Add text chunks and their embeddings to the ChromaDB collection."""
    
    # Generate embeddings
    embeddings = nomic_model.encode(text_chunks)
    embeddings = np.array(embeddings)  # Convert to NumPy array

    # Prepare metadata with the filename
    metadatas = [{"filename": filename} for _ in range(len(text_chunks))]

    # Add data to ChromaDB collection
    collection.add(
        ids=[str(uuid.uuid4()) for _ in range(len(text_chunks))],  # Unique IDs for each chunk
        documents=text_chunks,  # Searchable text content
        embeddings=embeddings.tolist(),  # Convert NumPy array to list for ChromaDB
        metadatas=metadatas  # Metadata with filename
    )

    print(f"Added {len(text_chunks)} chunks from {filename} to the collection.")

def retrieve_from_collection(query, top_k=5):
    """Retrieve the most relevant text chunks from ChromaDB based on the query."""
    
    # Convert query to embedding using the same model used for indexing
    query_embedding = nomic_model.encode([query])  # Convert query text to an embedding

    # Query ChromaDB using the precomputed embedding
    results = collection.query(
        query_embeddings=query_embedding.tolist(),  # Provide the query embedding
        n_results=top_k  # Number of relevant chunks to retrieve
    )

    # Extract documents from results
    retrieved_chunks = results['documents'][0] if results['documents'] else []
    return retrieved_chunks

