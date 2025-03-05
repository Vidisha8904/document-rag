import streamlit as st
from PyPDF2 import PdfReader
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from db import add_to_collection, retrieve_from_collection
from io import BytesIO
import camelot
import json
import traceback
import requests  # Add this for URL downloading
import time
import shutil

load_dotenv()
os.getenv("OPENAI_API_KEY")

# Initialize session state
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
            "source": pdf.name,  # You can keep this for JSON content if you want
            "text": text_data["text"],
            "tables": table_data["tables"]
        }
        json_content = json.dumps(combined_data, ensure_ascii=False, indent=2)
        
        doc = Document(
            page_content=json_content,
            metadata={"filename": pdf.name}  # Changed "source" to "filename"
        )
        documents.append(doc)
        
        st.sidebar.write(f"Processed: {pdf.name} - Found {table_count} tables and text extracted")

        with open("output.json", "w", encoding="utf-8") as json_file:
            json.dump(combined_data, json_file, ensure_ascii=False, indent=2)

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
        try:
            # Parse JSON content
            data = json.loads(doc.page_content)
        except json.JSONDecodeError:
            st.error(f"Error parsing JSON for {doc.metadata['filename']}")
            continue
        
        try:
            text_entries = data.get("text", [])
            table_entries = data.get("tables", [])
            
            # Process text entries
            if text_entries:
                text_content = "\n".join(entry.get("content", "") for entry in text_entries if entry.get("content"))
                
                if text_content:
                    text_chunks = text_splitter.create_documents(
                        texts=[text_content],
                        metadatas=[{
                            "source": doc.metadata["filename"], 
                            "type": "text"
                        }]
                    )
                    
                    chunk_texts = [chunk.page_content for chunk in text_chunks]
                    try:
                        add_to_collection(chunk_texts, doc.metadata['filename'])
                        all_chunks.extend(text_chunks)
                    except Exception as e:
                        st.error(f"Failed to add text chunks for {doc.metadata['filename']}: {e}")
            
            # Process table entries
            for table in table_entries:
                table_json = json.dumps(table, ensure_ascii=False)
                
                table_chunk = Document(
                    page_content=table_json,
                    metadata={
                        "source": doc.metadata["filename"], 
                        "type": "table"
                    }
                )
                
                try:
                    add_to_collection([table_json], doc.metadata['filename'])
                    all_chunks.append(table_chunk)
                except Exception as e:
                    st.error(f"Failed to add table chunk for {doc.metadata['filename']}: {e}")
            
            # Sidebar update
            total_chunks = len([c for c in all_chunks if c.metadata["source"] == doc.metadata["filename"]])
            st.sidebar.write(f"Created {total_chunks} chunks for {doc.metadata['filename']}")
        
        except Exception as e:
            st.error(f"Unexpected error processing {doc.metadata['filename']}: {e}")
    
    return all_chunks

def format_docs(docs, sources):
    return "\n\n".join(f"Source: {source}\n{doc_text}" for doc_text, source in zip(docs, sources))

def user_input(user_question):
    print(f"\n[Query] User question: {user_question}")
    
    # Get relevant documents using ChromaDB
    docs = retrieve_from_collection(user_question, top_k=8)
    print(f"[ChromaDB] Retrieved {len(docs) if isinstance(docs, list) else 0} documents")
    
    # Extract sources and texts
    if isinstance(docs, list) and docs:
        if isinstance(docs[0], dict) and 'metadata' in docs[0]:
            sources = [doc['metadata'].get('filename', 'Unknown') for doc in docs]  # Updated to use get() with default
            doc_texts = [doc['document'] for doc in docs]
        else:
            sources = ['Unknown'] * len(docs)
            doc_texts = docs
    else:
        sources = []
        doc_texts = []
    
    print("\n[Retrieved Documents]")
    for i, (source, text) in enumerate(zip(sources, doc_texts)):
        print(f"Document {i+1} from {source}")
        print(f"Preview: {text[:5000]}...")
    
    # Format context for GPT
    context = format_docs(doc_texts, sources)
    print(f"\n[Context for GPT] Length: {len(context)} characters")
    print(f"Preview: {context[:5000]}...")
    
    # Get model response
    model = ChatOpenAI(model="gpt-4", temperature=0.3)  # Fixed model name typo
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

# Rest of the code remains the same...
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

        url_input = st.text_area("Paste PDF URLs (one per line)", height=100)
        urls = [url.strip() for url in url_input.split('\n') if url.strip()]  # Split by line and filter empty
        
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                all_pdfs = []
                
                # Add uploaded PDFs
                if pdf_docs:
                    print("\n[Process Started] Processing PDF files...")
                    all_pdfs.extend(pdf_docs)
                
                # Download PDFs from URLs and add them
                if urls:
                    for url in urls:
                        pdf = download_pdf_from_url(url)
                        if pdf:
                            all_pdfs.append(pdf)
                
                if all_pdfs:   
                    documents = get_pdf_text(all_pdfs)  # Pass all_pdfs instead of pdf_docs
                    text_chunks = get_text_chunks(documents)
                    print("[Process Completed] PDF processing and chunking finished")
                    st.success("Done")
                else:
                    st.error("Please upload PDF files or provide valid PDF URLs first.")
    
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