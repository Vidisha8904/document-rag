# db.py
import chromadb
import uuid
from langchain_community.embeddings import HuggingFaceEmbeddings

# Initialize ChromaDB client
client = chromadb.HttpClient(host="localhost", port=8000)

# Create or get the collection
# collection = client.get_or_create_collection("allminilm-test-1")
# client.delete_collection("allminilm-test-1")
collection = client.get_or_create_collection("test-4", metadata={"hnsw:space": "cosine", "embedding_dimension": 768})
# Load embedding model
embedding_model = HuggingFaceEmbeddings(
    # model_name="BAAI/bge-large-en-v1.5",
    model_name="thenlper/gte-base",
    model_kwargs={'trust_remote_code': True}
)

def remove_existing_file_entries(filename):
    """Remove all existing entries for a given filename from the collection."""
    try:
        results = collection.get(where={"filename": filename})
        
        if results and results['ids']:
            collection.delete(ids=results['ids'])
            print(f"Removed {len(results['ids'])} existing entries for {filename}")
    except Exception as e:
        print(f"Error removing existing entries: {str(e)}")

def add_to_collection(text_chunks, filename):
    """Add text chunks and their embeddings to the ChromaDB collection."""
    try:
        remove_existing_file_entries(filename)  # Ensure duplicates are not added

        # Generate embeddings
        embeddings = embedding_model.embed_documents(text_chunks)  # Fix: Use embed_documents()

        # Prepare metadata with the filename
        metadatas = [{"filename": filename} for _ in range(len(text_chunks))]

        # Add data to ChromaDB collection
        collection.add(
            ids=[str(uuid.uuid4()) for _ in range(len(text_chunks))],  # Unique IDs for each chunk
            documents=text_chunks,  # Searchable text content
            embeddings=embeddings,  # Fix: embeddings is already a list, no need to convert
            metadatas=metadatas  # Metadata with filename
        )

        print(f"Added {len(text_chunks)} chunks from {filename} to the collection.")
    except Exception as e:
        print(f"Error adding chunks to collection: {str(e)}")
        raise

def retrieve_from_collection(query, top_k=8):
    """Retrieve the most relevant text chunks from ChromaDB based on the query."""
    try:
        # Convert query to embedding
        query_embedding = embedding_model.embed_query(query)  # Fix: Remove list brackets []

        # Query ChromaDB using the computed embedding
        results = collection.query(
            query_embeddings=[query_embedding],  # Fix: Wrap in a list
            n_results=top_k,  # Number of relevant chunks to retrieve
            include=['documents', 'metadatas','distances'],
              # Include both documents and metadata in results
        )

        print(123,results)        

        # Format results
        if results['documents'] and results['metadatas']:
            return [
                {'document': doc, 'metadata': meta}
                for doc, meta in zip(results['documents'][0], results['metadatas'][0])
            ]
        return []
        
    except Exception as e:
        print(f"Error retrieving from collection: {str(e)}")
        return []
