import streamlit as st
from PyPDF2 import PdfReader
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from keybert import KeyBERT
from db import add_to_collection, retrieve_from_collection, remove_existing_file_entries

load_dotenv()
os.getenv("OPENAI_API_KEY")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'user_input' not in st.session_state:
    st.session_state.user_input = ''
if 'submitted' not in st.session_state:
    st.session_state.submitted = False
if 'bm25_index' not in st.session_state:
    st.session_state.bm25_index = None
if 'bm25_corpus' not in st.session_state:
    st.session_state.bm25_corpus = []
if 'bm25_chunks' not in st.session_state:
    st.session_state.bm25_chunks = []

import re
from PyPDF2 import PdfReader
from langchain.docstore.document import Document


def get_pdf_text(pdf_docs):
    documents = []
    for pdf in pdf_docs:
        print(f"\n[PDF Processing] Started processing: {pdf.name}")
        pdf_reader = PdfReader(pdf)
        text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            text += page.extract_text() + "\n"
            print(f"[PDF Processing] Processed page {page_num + 1} of {pdf.name}")
        
        # Store complete text with metadata
        doc = Document(
            page_content=text,
            metadata={"filename": pdf.name}  # Changed from "source" to "filename"
        )
        documents.append(doc)
        print(f"[PDF Processing] Completed processing: {pdf.name}")
        st.sidebar.write(f"Processed: {pdf.name}")
    return documents

def get_text_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

    all_chunks = []
    total_chunks = 0
    st.session_state.bm25_corpus = []
    st.session_state.bm25_chunks = []

    for doc in documents:
        chunks = text_splitter.create_documents(
            texts=[doc.page_content],
            metadatas=[doc.metadata]
        )
        all_chunks.extend(chunks)
        # Extract text and add to ChromaDB with source information
        chunk_texts = [chunk.page_content for chunk in chunks]
        try:
            add_to_collection(chunk_texts, doc.metadata['filename'])  # Updated to use 'filename'
            print(f"[ChromaDB] Added {len(chunks)} chunks from {doc.metadata['filename']}")
            print(f"[ChromaDB] First chunk preview: {chunk_texts[0][:200]}...")
        except Exception as e:
            print(f"[ChromaDB] Error adding chunks: {str(e)}")
        
        total_chunks += len(chunks)
        st.sidebar.write(f"Created {len(chunks)} chunks for {doc.metadata['filename']}")
    
        print(f"\n[Summary] Total chunks created and stored in chromadb: {total_chunks}")

        # Prepare BM25 corpus
        for i, chunk in enumerate(chunks):
            cleaned_text = re.sub(r'[^\w\s]', '', chunk.page_content.lower())
            print(f"🧹 Cleaned Text (first 500 chars): {cleaned_text[:500]}")
            tokenized_chunk = cleaned_text.split()
            token_preview = " | ".join(tokenized_chunk[:30]) + ("..." if len(tokenized_chunk) > 30 else "")
            print(f"🔤 Tokenized Chunk Preview: {token_preview}")
            st.session_state.bm25_corpus.append(tokenized_chunk)
            st.session_state.bm25_chunks.append(chunk)

            print(f"🔹 Created Chunk {i+1} from {doc.metadata['filename']} (Characters: {len(chunk.page_content)}, Tokens: {len(tokenized_chunk)})")

    # Initialize BM25 index
    st.session_state.bm25_index = BM25Okapi(st.session_state.bm25_corpus)
    print("✅ BM25 index initialized.")

    print("📄 Stored BM25 Chunks:")
    # for i, chunk in enumerate(st.session_state.bm25_chunks):
    #     print(f"🔹 Chunk {i+1}: {chunk.page_content[:300]}...")  # Print first 300 characters
    #     print(f"   🔗 Source: {chunk.metadata.get('filename', 'Unknown')}")  # Updated source → filename
    #     print("-" * 80)

    return all_chunks


def format_docs(docs):
    formatted_docs = []
    for doc in docs:
        if hasattr(doc, 'metadata') and hasattr(doc, 'page_content'):
            # Document object from BM25
            source = doc.metadata.get('filename', 'Unknown')
            content = doc.page_content
        elif isinstance(doc, dict):
            # Dictionary from semantic search
            source = doc.get('metadata', {}).get('filename', 'Unknown')
            content = doc.get('document', '') if 'document' in doc else ''
        else:
            # Fallback
            source = 'Unknown'
            content = str(doc)
            
        formatted_docs.append(f"Source: {source}\n{content}")
    
    return "\n\n".join(formatted_docs)


def extract_keywords(query, top_n=3):
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(query, keyphrase_ngram_range=(1, 2,3 ,4 ), stop_words='english', top_n=top_n)
    extracted = [kw[0].lower() for kw in keywords] if keywords else query.lower().split()
    
    print(f"🔍 Extracted Keywords for Query: {query} → {extracted}")
    return extracted


def bm25_search(query, k=4):
    if st.session_state.bm25_index is None:
        return []

    keywords = extract_keywords(query)
    scores = st.session_state.bm25_index.get_scores(keywords)
    top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    results = [st.session_state.bm25_chunks[i] for i in top_k_indices]

    print("🔎 BM25 Search Results:")
    for i, doc in enumerate(results):
        print(f"📄 {i+1}. Source: {doc.metadata.get('filename', 'Unknown')} (Score: {scores[top_k_indices[i]]:.4f})")  # Updated source → filename
        print(f"   📜 Content: {doc.page_content[:500]}...")
        print("-" * 80)

    return results


# def reciprocal_rank_fusion(results_list, k=60):
#     """Apply Reciprocal Rank Fusion (RRF) to combine rankings"""
#     doc_scores = {}
#     doc_objects = {}  

#     for result_list in results_list:
#         for rank, doc in enumerate(result_list, start=1):
#             print("Type of doc:", type(doc))
#             if doc["document"] not in doc_scores:
#                 doc_scores[doc.page_content] = 0
#                 doc_objects[doc.page_content] = doc  
            
#             doc_scores[doc.page_content] += 1 / (k + rank)

#     sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
#     print("✅ RRF Applied - Ranking Adjusted.")
#     return [doc_objects[doc[0]] for doc in sorted_docs]  

def reciprocal_rank_fusion(results_list, k=60):
    """Apply Reciprocal Rank Fusion (RRF) to combine rankings"""
    doc_scores = {}
    doc_objects = {}  

    for result_list in results_list:
        for rank, doc in enumerate(result_list, start=1):
            # Handle different result types
            if isinstance(doc, dict) and "document" in doc:
                # For semantic search results
                doc_text = doc["document"]
                doc_obj = doc
            else:
                # For BM25 results (Document objects)
                doc_text = doc.page_content
                doc_obj = doc
            
            if doc_text not in doc_scores:
                doc_scores[doc_text] = 0
                doc_objects[doc_text] = doc_obj
            
            doc_scores[doc_text] += 1 / (k + rank)

    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    print("✅ RRF Applied - Ranking Adjusted.")
    
    # Return the objects in the new order
    return [doc_objects[doc[0]] for doc in sorted_docs]

# Changes made: source → filename
def hybrid_search(query, k=4):
    print(f"🔎 Searching for: {query}")
    semantic_results = retrieve_from_collection(query, top_k=8)

    print("✅ Semantic Search Results:")
    for i, doc in enumerate(semantic_results):
        # Check document structure and access fields accordingly
        if hasattr(doc, 'metadata') and hasattr(doc, 'page_content'):
            source = doc.metadata.get('filename', 'Unknown')
            preview = doc.page_content[:200]
        elif isinstance(doc, dict):
            source = doc.get('metadata', {}).get('filename', 'Unknown')
            preview = doc.get('document', '')[:200] if 'document' in doc else ''
        else:
            source = 'Unknown'
            preview = str(doc)[:200] if doc else ''
            
        print(f"📄 {i+1}. Source: {source} (Preview: {preview})")

    bm25_results = bm25_search(query, k=k)

    combined_results = reciprocal_rank_fusion([semantic_results, bm25_results])

    print("✅ Combined Search Results (RRF Applied):")
    for i, doc in enumerate(combined_results[:k]):
        # Check document type and access fields accordingly
        if hasattr(doc, 'metadata') and hasattr(doc, 'page_content'):
            # Document object from BM25
            source = doc.metadata.get('filename', 'Unknown')
            preview = doc.page_content[:200]
        elif isinstance(doc, dict):
            # Dictionary from semantic search
            source = doc.get('metadata', {}).get('filename', 'Unknown')
            preview = doc.get('document', '')[:200] if 'document' in doc else ''
        else:
            # Fallback for unknown types
            source = 'Unknown'
            preview = str(doc)[:200] if doc else ''
            
        print(f"📄 {i+1}. Source: {source} (Preview: {preview})")

    return combined_results[:k]

def user_input(user_question):
    docs = hybrid_search(user_question, k=10)

    print("📥 Retrieved Documents:")
    sources = []
    for doc in docs:
        if hasattr(doc, 'metadata'):
            source = doc.metadata.get('filename', 'Unknown')
        elif isinstance(doc, dict):
            source = doc.get('metadata', {}).get('filename', 'Unknown')
        else:
            source = 'Unknown'
        
        sources.append(source)
        print(f"📄 From: {source}")

    sources = list(set(sources))
    sources_str = ', '.join(sources)

    context = format_docs(docs)
    print(f"✅ Context prepared for GPT (Sources: {sources_str})")
    
    # The rest of the function remains unchanged

    messages = [
        {
            "role": "system",
            "content": f"""You are an AI assistant that helps users understand PDF documents. 
            The following information comes from these PDF files: {sources_str}
            **IMPORTANT: Response Format**  
            - If you find relevant information: **"Sources: [list of PDF filenames, comma-separated]"**  
            - If you don't find relevant information: **"No relevant information found in the provided PDFs."**  
            - After stating the sources, provide a detailed and structured answer.

            **Rules for Answering:**  
            - Use **only** the provided context to generate responses.   
            - If the answer is **not available**, state: **"Answer is not available in the context."** Do not generate speculative or misleading answers.  
            - If the query involves **calculations**, perform them and provide the exact result.    
            - If the query is **unclear**, ask for clarification instead of making assumptions.  

            **Conversation Memory:**  
            - Support **multi-turn conversations** by remembering previous interactions.  
            - Reference prior interactions when relevant to maintain consistency.   

            Maintain an appropriate tone—**formal, conversational, concise, or elaborate**, depending on the query.  
            """
        },
        {
            "role": "user",
            "content": f"""Question: {user_question}

            Information from PDFs:
            {format_docs(docs)}"""
        }
    ]

    model = ChatOpenAI(model="gpt-4o", temperature=0.3)
    response = model.invoke(messages)

    print(f"🤖 GPT Response: {response.content[:300]}...")
    return response.content


def display_chat_message(role, content):
    with st.container():
        if role == "user":
            st.markdown(f"""
            <div style="display: flex; justify-content: flex-end;">
                <div style="background-color: #007AFF; color: white; padding: 10px; 
                border-radius: 15px; margin: 5px; max-width: 70%;">
                     {content}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="display: flex; justify-content: flex-start;">
                <div style="background-color: #E9ECEF; color: black; padding: 10px; 
                border-radius: 15px; margin: 5px; max-width: 70%;">
                    🤖 {content}
                </div>
            </div>
            """, unsafe_allow_html=True)


def handle_submit():
    if st.session_state.user_input and not st.session_state.submitted:
        user_question = st.session_state.user_input
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        response = user_input(user_question)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.session_state.user_input = ''
        st.session_state.submitted = True

def main():
    st.set_page_config("Chat PDF", layout="wide")
    st.header("Chat with PDF - Hybrid Search with chromadb (Semantic + BM25 copy)")
    
    # Create debug container in sidebar
    st.session_state.debug_container = st.sidebar

    # Main sidebar content
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", 
                                  accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                if pdf_docs:

                    print("\n[Process Started] Processing PDF files...")
                    for pdf in pdf_docs:
                        remove_existing_file_entries(pdf.name)
                    documents = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(documents)
                
                    st.success("Processing complete! Vector store and BM25 index created.")
                else:
                    st.error("Please upload PDF files first.")
    # Chat interface
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            display_chat_message(message["role"], message["content"])

    col1, col2 = st.columns([6, 1])
    with col1:
        st.text_input("Ask any Question from the PDF Files", 
                     key="user_input", 
                     on_change=handle_submit)
    
    if not st.session_state.user_input:
        st.session_state.submitted = False

if __name__ == "__main__":
    main()
