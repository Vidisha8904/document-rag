import streamlit as st
from PyPDF2 import PdfReader
import os
import pickle
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()
os.getenv("OPENAI_API_KEY")

# Initialize session state for chat history and submitted flag
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'user_input' not in st.session_state:
    st.session_state.user_input = ''
if 'submitted' not in st.session_state:
    st.session_state.submitted = False

def get_pdf_text(pdf_docs):
    print("\n=== Starting PDF Text Extraction ===")
    documents = []
    for pdf in pdf_docs:
        print(f"\nProcessing PDF: {pdf.name}")
        pdf_reader = PdfReader(pdf)
        text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            text += page.extract_text() + "\n"
            print(f"Page {page_num + 1} extracted")
        
        doc = Document(
            page_content=text,
            metadata={"source": pdf.name}
        )
        documents.append(doc)
        print(f"Completed processing: {pdf.name}")
    
    print(f"\nTotal documents processed: {len(documents)}")
    return documents

def get_text_chunks(documents):
    print("\n=== Starting Text Chunking ===")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
    )
    
    all_chunks = []
    for doc in documents:
        print(f"\nChunking document: {doc.metadata['source']}")
        chunks = text_splitter.create_documents(
            texts=[doc.page_content],
            metadatas=[doc.metadata]
        )
        all_chunks.extend(chunks)
        print(f"Created {len(chunks)} chunks from {doc.metadata['source']}")
    
    print(f"\nTotal chunks created: {len(all_chunks)}")
    return all_chunks

def get_vector_store(chunks):
    print("\n=== Creating Vector Store ===")
    embeddings = HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1",
        model_kwargs={'trust_remote_code': True}
    )
    
    print(f"Processing {len(chunks)} chunks for embedding...")
    vector_store = FAISS.from_documents(chunks, embedding=embeddings)
    
    vector_store.save_local("faiss_index")
    print("Vector store saved to faiss_index")
    print(f"Vector store size: {len(chunks)} vectors")

def format_docs(docs):
    print("\n=== Formatting Documents for Context ===")
    print(f"Formatting {len(docs)} documents")
    return "\n\n".join(f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}" for doc in docs)

def user_input(user_question):
    print("\n=== Processing User Query ===")
    print(f"Query received: {user_question}")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1",
        model_kwargs={'trust_remote_code': True}
    )
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    print("\nSearching for relevant documents...")
    docs = new_db.similarity_search(user_question, k=4)
    
    print("\nRetrieved Documents:")
    for i, doc in enumerate(docs, 1):
        print(f"\nDocument {i}:")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"Content Preview: {doc.page_content[:200]}...")
    
    sources = list(set(doc.metadata.get('source', 'Unknown') for doc in docs))
    sources_str = ', '.join(sources)
    print(f"\nSources being used: {sources_str}")
    
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
    
    print("\nSending query to GPT...")
    model = ChatOpenAI(model="gpt-4", temperature=0.3)
    response = model.invoke(messages)
    print("\nResponse received from GPT")
    print("Response preview:", response.content[:200], "...")
    
    return response.content

def handle_submit():
    if st.session_state.user_input and not st.session_state.submitted:
        user_question = st.session_state.user_input
        print("\n=== New Question Processing Started ===")
        
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        response = user_input(user_question)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        st.session_state.user_input = ''
        st.session_state.submitted = True
        print("=== Question Processing Completed ===\n")

def display_faiss_contents():
    print("\n=== Displaying FAISS Index Contents ===")
    from langchain.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1",
        model_kwargs={'trust_remote_code': True}
    )
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    documents = vector_store.docstore._dict.values()
    num_vectors = len(vector_store.docstore._dict)
    embeddings_array = vector_store.index.reconstruct_n(0, num_vectors)
    
    print(f"Total vectors in store: {num_vectors}")
    print(f"Embedding dimensions: {embeddings_array.shape}")
    print("=== FAISS Index Display Completed ===\n")

    return vector_store

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

def main():
    st.set_page_config("Chat PDF", layout="wide")
    st.header("Chat with PDF - nomic and gpt")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", 
                                  accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                if pdf_docs:
                    print("\n=== Starting PDF Processing Pipeline ===")
                    documents = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(documents)
                    st.write(text_chunks)
                    get_vector_store(text_chunks)
                    st.success("Done")
                    print("=== PDF Processing Pipeline Completed ===\n")
                else:
                    st.error("Please upload PDF files first.")
                    print("Error: No PDF files uploaded")

    if st.button("View FAISS Index Contents"):
        display_faiss_contents()

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