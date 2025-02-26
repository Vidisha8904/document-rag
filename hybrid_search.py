# working fine no  error
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
        pdf_reader = PdfReader(pdf)
        text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            extracted_text = page.extract_text()        
        doc = Document(
            page_content=extracted_text,
            metadata={"source": pdf.name}
        )
        documents.append(doc)
        print(f"✅ Processed PDF: {pdf.name} (Character count: {len(text)})")
    
    return documents


def get_text_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    
    all_chunks = []
    st.session_state.bm25_corpus = []
    st.session_state.bm25_chunks = []
    
    for doc in documents:
        chunks = text_splitter.create_documents(
            texts=[doc.page_content],
            metadatas=[doc.metadata]
        )
        all_chunks.extend(chunks)
        
        # Prepare BM25 corpus
        for i, chunk in enumerate(chunks):
            tokenized_chunk = chunk.page_content.lower().split()
            st.session_state.bm25_corpus.append(tokenized_chunk)
            st.session_state.bm25_chunks.append(chunk)
            
            print(f"🔹 Created Chunk {i+1} from {doc.metadata['source']} (Characters: {len(chunk.page_content)}, Tokens: {len(tokenized_chunk)})")
    
    # Initialize BM25 index
    st.session_state.bm25_index = BM25Okapi(st.session_state.bm25_corpus)
    print("✅ BM25 index initialized.")
    print("#####################################################################################################")

    print(st.session_state.bm25_corpus[0])

    print("📄 Stored BM25 Chunks:")
    for i, chunk in enumerate(st.session_state.bm25_chunks):
        print(f"🔹 Chunk {i+1}: {chunk.page_content[:300]}...")  # Print first 300 characters
        print(f"   🔗 Source: {chunk.metadata.get('source', 'Unknown')}")
        print("-" * 80)

    
    return all_chunks

def get_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1",
        model_kwargs={'trust_remote_code': True}
    )
    vector_store = FAISS.from_documents(chunks, embedding=embeddings)
    
    print(f"✅ Vector store created with {len(chunks)} chunks (Model: nomic-ai/nomic-embed-text-v1)")
    
    vector_store.save_local("faiss_index")

def extract_keywords(query, top_n=3):
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(query, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=top_n)
    extracted = [kw[0].lower() for kw in keywords] if keywords else query.lower().split()
    
    print(f"🔍 Extracted Keywords for Query: {query} → {extracted}")
    return extracted

def bm25_search(query, k=4):
    if st.session_state.bm25_index is None:
        return []
    
    keywords = extract_keywords(query)
    print(keywords)
    # scores = st.session_state.bm25_index.get_scores(query.lower().split())
    scores = st.session_state.bm25_index.get_scores(keywords)
    top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    results = [st.session_state.bm25_chunks[i] for i in top_k_indices]
    
    print("1234567890     🔎 BM25 Search Results:")
    for i, doc in enumerate(results):
        print(f"📄 {i+1}. Source: {doc.metadata.get('source', 'Unknown')} (Score: {scores[top_k_indices[i]]:.4f})")
        print(f"   📜 Content: {doc.page_content[:500]}...")  # Show first 500 characters of content
        print("-" * 80) 
    
    return results

def reciprocal_rank_fusion(results_list, k=60):
    """Apply Reciprocal Rank Fusion (RRF) to combine rankings"""
    doc_scores = {}
    doc_objects = {}  

    for result_list in results_list:
        for rank, doc in enumerate(result_list, start=1):
            if doc.page_content not in doc_scores:
                doc_scores[doc.page_content] = 0
                doc_objects[doc.page_content] = doc  
            
            doc_scores[doc.page_content] += 1 / (k + rank)

    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    print("✅ RRF Applied - Ranking Adjusted.")
    return [doc_objects[doc[0]] for doc in sorted_docs]  

def hybrid_search(query, k=4):
    print(f"🔎 Searching for: {query}")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1",
        model_kwargs={'trust_remote_code': True}
    )
    vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    semantic_results = vector_db.similarity_search(query, k=k)
    
    print("✅ Semantic Search Results:")
    for i, doc in enumerate(semantic_results):
        print(f"📄 {i+1}. Source: {doc.metadata.get('source', 'Unknown')} (Preview: {doc.page_content[:200]})")
    
    bm25_results = bm25_search(query, k=k)
    
    combined_results = reciprocal_rank_fusion([semantic_results, bm25_results])
    
    print("✅ Combined Search Results (RRF Applied):")
    for i, doc in enumerate(combined_results[:k]):  
        print(f"📄 {i+1}. Source: {doc.metadata.get('source', 'Unknown')} (Preview: {doc.page_content[:200]})")
    
    return combined_results[:k]

def format_docs(docs):
    return "\n\n".join(f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}" for doc in docs)

def user_input(user_question):
    docs = hybrid_search(user_question, k=10)
    
    print("📥 Retrieved Documents:")
    for doc in docs:
        print(f"📄 From: {doc.metadata.get('source', 'Unknown')}")
    
    sources = list(set(doc.metadata.get('source', 'Unknown') for doc in docs))
    sources_str = ', '.join(sources)
    
    context = format_docs(docs)
    print(f"✅ Context prepared for GPT (Sources: {sources_str})")
    
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
    st.header("Chat with PDF - Hybrid Search (Semantic + BM25 copy)")
    
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
                    documents = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(documents)
                    get_vector_store(text_chunks)
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
