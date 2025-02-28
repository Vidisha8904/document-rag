import streamlit as st
from PyPDF2 import PdfReader
import os
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from io import BytesIO
import camelot
import json
import traceback
import requests  # Add this for URL downloading
import time
import shutil

load_dotenv()
os.getenv("OPENAI_API_KEY")

# Initialize session state for chat history and submitted flag
if 'session_start_time' not in st.session_state:
    if os.path.exists("faiss_index"):
        try:
            shutil.rmtree("faiss_index")
            print("Cleared data from previous browser session")
        except Exception as e:
            print(f"Error clearing previous session data: {str(e)}")
    st.session_state.session_start_time = time.time()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'user_input' not in st.session_state:
    st.session_state.user_input = ''
if 'submitted' not in st.session_state:
    st.session_state.submitted = False

def download_pdf_from_url(url):
    """Download a PDF from a URL and return it as a BytesIO object."""
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()  # Raise an error for bad status codes
        if 'application/pdf' not in response.headers.get('Content-Type', ''):
            st.sidebar.write(f"Warning: {url} does not appear to be a PDF.")
            return None
        pdf_bytes = BytesIO(response.content)
        pdf_bytes.name = url.split('/')[-1] or "downloaded_pdf.pdf"  # Set a default name if none found
        return pdf_bytes
    except Exception as e:
        st.sidebar.write(f"Error downloading PDF from {url}: {str(e)}")
        return None

def get_pdf_text(pdf_docs):
    documents = []
    for pdf in pdf_docs:
        # Save to temporary file for camelot (since it needs a file path)
        temp_path = f"temp_{pdf.name}"
        with open(temp_path, "wb") as f:
            f.write(pdf.read())
        
        # Extract text with PyPDF2 
        text_data = {"text": []}
        try:
            pdf_reader = PdfReader(temp_path)
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text().strip()
                if page_text:
                    text_data["text"].append({"page": page_num + 1, "content": page_text})
        except Exception as e:
            st.sidebar.write(f"Error extracting text from {pdf.name}: {str(e)}")
        
        # Extract tables with camelot
        table_data = {"tables": []}
        table_count = 0
        try:
            tables = camelot.read_pdf(temp_path, pages='all', flavor='lattice')
            for i, table in enumerate(tables):
                if table.df.size > 0:
                    table_dict = {
                        "table_id": f"Table {i + 1}",
                        "page": table.parsing_report['page'],
                        "data": table.df.replace({None: ""}).to_dict(orient='records')
                    }
                    table_data["tables"].append(table_dict)
                    table_count += 1
            if table_count == 0:
                tables = camelot.read_pdf(temp_path, pages='all', flavor='stream')
                for i, table in enumerate(tables):
                    if table.df.size > 0:
                        table_dict = {
                            "table_id": f"Table {i + 1}",
                            "page": table.parsing_report['page'],
                            "data": table.df.replace({None: ""}).to_dict(orient='records')
                        }
                        table_data["tables"].append(table_dict)
                        table_count += 1
        except Exception as e:
            st.sidebar.write(f"Error extracting tables from {pdf.name}: {str(e)}")
        
        combined_data = {
            "source": pdf.name,
            "text": text_data["text"],
            "tables": table_data["tables"]
        }
        json_content = json.dumps(combined_data, ensure_ascii=False, indent=2)
        
        doc = Document(
            page_content=json_content,
            metadata={"source": pdf.name}
        )
        documents.append(doc)
        
        st.sidebar.write(f"Processed: {pdf.name} - Found {table_count} tables and text extracted")

        # with open("output.json", "w", encoding="utf-8") as json_file:
        #     json.dump(combined_data, json_file, ensure_ascii=False, indent=2)

        os.remove(temp_path)
    
    return documents

def get_text_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=250,
        separators=["\n\n", "\n", ","]
    )
    
    all_chunks = []
    for doc in documents:
        data = json.loads(doc.page_content)
        text_entries = data.get("text", [])
        table_entries = data.get("tables", [])
        
        text_content = "\n".join(entry.get("content", "") for entry in text_entries if entry.get("content"))
        if text_content:
            text_chunks = text_splitter.create_documents(
                texts=[text_content],
                metadatas=[{"source": doc.metadata["source"], "type": "text"}]
            )
            all_chunks.extend(text_chunks)
        
        for table in table_entries:
            table_json = json.dumps(table, ensure_ascii=False)
            table_chunk = Document(
                page_content=table_json,
                metadata={"source": doc.metadata["source"], "type": "table"}
            )
            all_chunks.append(table_chunk)
        
        total_chunks = len([c for c in all_chunks if c.metadata["source"] == doc.metadata["source"]])
        text_chunk_count = len([c for c in all_chunks if c.metadata["source"] == doc.metadata["source"] and c.metadata["type"] == "text"])
        table_chunk_count = len(table_entries)
        st.sidebar.write(f"Created {total_chunks} chunks for {doc.metadata['source']} "
                        f"({text_chunk_count} text, {table_chunk_count} tables)")
    
    return all_chunks

def get_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1",
        model_kwargs={'trust_remote_code': True}
    )
    vector_store = FAISS.from_documents(chunks, embedding=embeddings)
    st.sidebar.write(f"Created vector store with {len(chunks)} chunks")
    vector_store.save_local("faiss_index")

def user_input(user_question):
    if not os.path.exists("faiss_index") or not os.path.exists("faiss_index/index.faiss"):
        return "Please upload or process PDF files/URLs first before asking questions."
    
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="nomic-ai/nomic-embed-text-v1",
            model_kwargs={'trust_remote_code': True}
        )
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question, k=6)
        
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source', 'Unknown')
            content_snippet = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
        
        sources = list(set(doc.metadata.get('source', 'Unknown') for doc in docs))
        sources_str = ', '.join(sources)
        
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
                
                Information from PDFs (in JSON format):
                {format_docs(docs)}"""
            }
        ]

        model = ChatOpenAI(model="gpt-4o", temperature=0.3)
        response = model.invoke(messages)
        return response.content
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(error_msg)
        return "An error occurred while processing your question. Please make sure you've uploaded or processed PDF files/URLs first."

def format_docs(docs):
    return "\n\n".join(f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}" for doc in docs)

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
    st.header("Chat with PDF")

    with st.sidebar:
        st.title("Menu:")
        
        # File uploader for PDFs
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        
        # Text area for URLs
        url_input = st.text_area("Paste PDF URLs (one per line)", height=100)
        urls = [url.strip() for url in url_input.split('\n') if url.strip()]  # Split by line and filter empty
        
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                all_pdfs = []
                
                # Add uploaded PDFs
                if pdf_docs:
                    all_pdfs.extend(pdf_docs)
                
                # Download PDFs from URLs
                if urls:
                    for url in urls:
                        pdf = download_pdf_from_url(url)
                        if pdf:
                            all_pdfs.append(pdf)
                
                if all_pdfs:
                    documents = get_pdf_text(all_pdfs)
                    print(documents)
                    text_chunks = get_text_chunks(documents)
                    get_vector_store(text_chunks)
                    st.success("Done")
                else:
                    st.error("Please upload PDF files or provide valid PDF URLs first.")
    
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            display_chat_message(message["role"], message["content"])

    if os.path.exists("faiss_index") and os.path.exists("faiss_index/index.faiss"):
        col1, col2 = st.columns([6, 1])
        with col1:
            st.text_input("Ask any Question from the PDF Files", 
                        key="user_input", 
                        on_change=handle_submit)
    else:
        st.warning("Please upload or process PDF files/URLs before asking questions.")
        st.text_input("Ask any Question from the PDF Files", 
                    key="user_input", 
                    on_change=handle_submit,
                    disabled=False)
    
    if not st.session_state.user_input:
        st.session_state.submitted = False

if __name__ == "__main__":
    main()