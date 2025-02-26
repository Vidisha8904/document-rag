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

load_dotenv()
os.getenv("OPENAI_API_KEY")

# Initialize session state for chat history and submitted flag
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'user_input' not in st.session_state:
    st.session_state.user_input = ''
if 'submitted' not in st.session_state:
    st.session_state.submitted = False




# [Rest of your code]

import pdfplumber
from io import BytesIO
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
import streamlit as st

import pdfplumber
from io import BytesIO
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
import streamlit as st
import traceback

# def get_pdf_text(pdf_docs):
#     documents = []
#     for pdf in pdf_docs:
#         pdf_file = BytesIO(pdf.read())
        
#         # Extract regular text with PyPDF2
#         pdf_reader = PdfReader(pdf_file)
#         text = ""
#         for page_num, page in enumerate(pdf_reader.pages):
#             text += page.extract_text() + "\n"
        
#         # Reset file pointer for pdfplumber
#         pdf_file.seek(0)
        
#         # Extract tables with pdfplumber
#         table_text = ""
#         table_count = 0
#         try:
#             if pdf_file.getbuffer().nbytes == 0:
#                 raise ValueError("Empty PDF file")
#             with pdfplumber.open(pdf_file) as pdf_doc:  # Rename to avoid confusion
#                 if not pdf_doc.pages:
#                     raise ValueError("PDF has no pages")
#                 for i, page in enumerate(pdf_doc.pages):
#                     tables = page.extract_tables() or []  # Default to empty list if None
#                     for j, table in enumerate(tables):
#                         if table:  # Check if table exists
#                             table_str = "\n".join(
#                                 ["|".join(cell if cell is not None else "" for cell in row) 
#                                  for row in table if row]
#                             )
#                             table_text += f"\nTable {i + 1}.{j + 1}:\n{table_str}\n"
#                             table_count += 1
#         except Exception as e:
#             st.sidebar.write(f"Error extracting tables from {pdf.name}: {str(e)}")
#             st.sidebar.write(f"Exception type: {type(e).__name__}")
#             st.sidebar.write(f"Traceback: {traceback.format_exc()}")
#             table_text = "No tables extracted due to an error."
        
#         # Combine regular text and table text
#         combined_text = text + "\n\nExtracted Tables:\n" + table_text
        
#         doc = Document(
#             page_content=combined_text,
#             metadata={"source": pdf.name}
#         )
#         documents.append(doc)
        
#         st.sidebar.write(f"Processed: {pdf.name} - Found {table_count} tables")
    
#     return documents



import camelot
import pdfplumber  # For better text extraction if needed, or use PyPDF2
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
import streamlit as st
import os
import traceback

def get_pdf_text(pdf_docs):
    documents = []
    for pdf in pdf_docs:
        # Save to temporary file for camelot (since it needs a file path)
        temp_path = f"temp_{pdf.name}"
        with open(temp_path, "wb") as f:
            f.write(pdf.read())
        
        # Extract text with PyPDF2 (or pdfplumber for better layout)
        text = ""
        try:
            pdf_reader = PdfReader(temp_path)
            for page_num, page in enumerate(pdf_reader.pages):
                text += page.extract_text() + "\n"
        except Exception as e:
            st.sidebar.write(f"Error extracting text from {pdf.name}: {str(e)}")
            text = "No text extracted due to an error."
        
        # Extract tables with camelot
        table_text = ""
        table_count = 0
        try:
            # Try 'lattice' flavor first (for bordered tables)
            tables = camelot.read_pdf(temp_path, pages='all', flavor='lattice')
            for i, table in enumerate(tables):
                if table.df.size > 0:  # Check if table is non-empty
                    table_text += f"\nTable {i + 1}:\n{table.df.to_markdown(index=False)}\n"
                    table_count += 1
            
            # If no tables found, try 'stream' flavor (for borderless tables)
            if table_count == 0:
                tables = camelot.read_pdf(temp_path, pages='all', flavor='stream')
                for i, table in enumerate(tables):
                    if table.df.size > 0:
                        table_text += f"\nTable {i + 1}:\n{table.df.to_markdown(index=False)}\n"
                        table_count += 1
        except Exception as e:
            st.sidebar.write(f"Error extracting tables from {pdf.name}: {str(e)}")
            st.sidebar.write(f"Exception type: {type(e).__name__}")
            st.sidebar.write(f"Traceback: {traceback.format_exc()}")
            table_text = "No tables extracted due to an error."
        
        # Combine regular text and table text
        combined_text = text
        if table_text:
            combined_text += "\n\nExtracted Tables:\n" + table_text
        
        doc = Document(
            page_content=combined_text,
            metadata={"source": pdf.name}
        )
        documents.append(doc)
        
        st.sidebar.write(f"Processed: {pdf.name} - Found {table_count} tables and text extracted")
        
        # Clean up temporary file
        os.remove(temp_path)
    
    return documents

def get_text_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    
    all_chunks = []
    for doc in documents:
        chunks = text_splitter.create_documents(
            texts=[doc.page_content],
            metadatas=[doc.metadata]
        )
        all_chunks.extend(chunks)
        
        # Debug print
        st.sidebar.write(f"Created {len(chunks)} chunks for {doc.metadata['source']}")
    
    return all_chunks

def get_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1",
        model_kwargs={'trust_remote_code': True}
    )
    vector_store = FAISS.from_documents(chunks, embedding=embeddings)
    
    # Debug print
    st.sidebar.write(f"Created vector store with {len(chunks)} chunks")
    
    vector_store.save_local("faiss_index")

def format_docs(docs):
    return "\n\n".join(f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}" for doc in docs)

def user_input(user_question):
    embeddings = HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1",
        model_kwargs={'trust_remote_code': True}
    )
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    # Get relevant documents
    docs = new_db.similarity_search(user_question, k=4)
    
    # Debug print
    st.sidebar.write("Retrieved Documents:")
    for doc in docs:
        st.sidebar.write(f"- From: {doc.metadata.get('source', 'Unknown')}")
    
    # Extract unique sources
    sources = list(set(doc.metadata.get('source', 'Unknown') for doc in docs))
    sources_str = ', '.join(sources)
    
    # Create a custom system message that forces mention of sources
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

    
    # Use ChatOpenAI directly for more control
    model = ChatOpenAI(model="gpt-4o", temperature=0.3)
    response = model.invoke(messages)
    
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
        
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        
        # Get AI response
        response = user_input(user_question)
        
        # Add AI response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Clear the input and set submitted flag
        st.session_state.user_input = ''
        st.session_state.submitted = True

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
                    documents = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(documents)
                    st.write(text_chunks)
                    get_vector_store(text_chunks)
                    st.success("Done")
                else:
                    st.error("Please upload PDF files first.")
    
    # Create a container for the chat history
    chat_container = st.container()

    # Display chat history
    with chat_container:
        for message in st.session_state.chat_history:
            display_chat_message(message["role"], message["content"])

    # Create the input box at the bottom with a submit button
    col1, col2 = st.columns([6, 1])
    with col1:
        st.text_input("Ask any Question from the PDF Files", 
                     key="user_input", 
                     on_change=handle_submit)
    
    # Reset submitted flag when the input is empty
    if not st.session_state.user_input:
        st.session_state.submitted = False

if __name__ == "__main__":
    main()