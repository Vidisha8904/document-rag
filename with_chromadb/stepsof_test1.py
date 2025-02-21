import streamlit as st
from PyPDF2 import PdfReader
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.docstore.document import Document
from dotenv import load_dotenv
from db import add_to_collection, retrieve_from_collection

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
            metadata={"source": pdf.name}
        )
        documents.append(doc)
        print(f"[PDF Processing] Completed processing: {pdf.name}")
        st.sidebar.write(f"Processed: {pdf.name}")
    return documents

def get_text_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    
    all_chunks = []
    total_chunks = 0
    
    for doc in documents:
        print(f"\n[Chunking] Processing document: {doc.metadata['source']}")
        chunks = text_splitter.create_documents(
            texts=[doc.page_content],
            metadatas=[doc.metadata]
        )
        all_chunks.extend(chunks)
        
        # Extract text and add to ChromaDB with source information
        chunk_texts = [chunk.page_content for chunk in chunks]
        try:
            add_to_collection(chunk_texts, doc.metadata['source'])
            print(f"[ChromaDB] Added {len(chunks)} chunks from {doc.metadata['source']}")
            print(f"[ChromaDB] First chunk preview: {chunk_texts[0][:200]}...")
        except Exception as e:
            print(f"[ChromaDB] Error adding chunks: {str(e)}")
        
        total_chunks += len(chunks)
        st.sidebar.write(f"Created {len(chunks)} chunks for {doc.metadata['source']}")
    
    print(f"\n[Summary] Total chunks created and stored: {total_chunks}")
    return all_chunks

def format_docs(docs, sources):
    return "\n\n".join(f"Source: {source}\n{doc_text}" for doc_text, source in zip(docs, sources))

def user_input(user_question):
    print(f"\n[Query] User question: {user_question}")
    
    # Get relevant documents using ChromaDB
    docs = retrieve_from_collection(user_question, top_k=4)
    print(f"[ChromaDB] Retrieved {len(docs) if isinstance(docs, list) else 0} documents")
    
    # Extract sources and texts
    if isinstance(docs, list) and docs and 'metadata' in docs[0]:
        sources = [doc['metadata']['filename'] for doc in docs]
        doc_texts = [doc['document'] for doc in docs]
    else:
        sources = ['Unknown'] * len(docs)
        doc_texts = docs
    
    print("\n[Retrieved Documents]")
    for i, (source, text) in enumerate(zip(sources, doc_texts)):
        print(f"Document {i+1} from {source}")
        print(f"Preview: {text[:200]}...")
    
    # Format context for GPT
    context = format_docs(doc_texts, sources)
    print(f"\n[Context for GPT] Length: {len(context)} characters")
    print(f"Preview: {context[:500]}...")
    
    # Get model response
    model = ChatOpenAI(model="gpt-4o", temperature=0.3)
    messages = [
        {
            "role": "system",
            "content": f"""You are an AI assistant that helps users understand PDF documents. 
            The following information comes from these PDF files: {', '.join(set(sources))}
            
            **IMPORTANT: Response Format**  
            - If you find relevant information: **"Sources: [list of PDF filenames, comma-separated]"**  
            - If you don't find relevant information: **"No relevant information found in the provided PDFs."**  
            - After stating the sources, provide a detailed and structured answer.

            **Rules for Answering:**  
            - Use **only** the provided context to generate responses.   
            - If the answer is **not available**, state: **"Answer is not available in the context."**
            - If the query involves **calculations**, perform them and provide the exact result.    
            - If the query is **unclear**, ask for clarification instead of making assumptions.  
            """
        },
        {
            "role": "user",
            "content": f"Question: {user_question}\n\nInformation from PDFs:\n{context}"
        }
    ]
    
    response = model.invoke(messages)
    print(f"\n[GPT Response]\n{response.content}\n")
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
    st.header("Chat with PDF - ChromaDB and GPT")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", 
                                  accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                if pdf_docs:
                    print("\n[Process Started] Processing PDF files...")
                    documents = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(documents)
                    print("[Process Completed] PDF processing and chunking finished")
                    st.success("Done")
                else:
                    st.error("Please upload PDF files first.")
    
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