import streamlit as st
from PyPDF2 import PdfReader
from io import BytesIO 
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# load_dotenv()
# os.getenv("OPENAI_API_KEY")

def get_pdf_text(pdf_docs):
    pdf_texts = []

    for pdf in pdf_docs:
        text = ""
        pdf_reader = PdfReader(BytesIO(pdf.read()))  # Convert to file-like object
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:  # Avoid NoneType errors
                text += page_text + "\n"
        
        pdf_texts.append((text, pdf.name))  # Store content & filename
    
    return pdf_texts    # List of tuples (text, filename)

def get_text_chunks(pdf_texts):
    """Each PDF is stored as one chunk, preserving its filename."""
    chunks = []
    metadata = []
    
    for text, filename in pdf_texts:
        chunks.append(text)  # Store the entire PDF content as one chunk
        metadata.append({"filename": filename})  # Associate metadata with the chunk
    
    return chunks, metadata


def get_vector_store(text_chunks, metadata):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    # Store texts with metadata in FAISS
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings, metadatas=metadata)
    vector_store.save_local("faiss_index")

    print(f"Stored {len(text_chunks)} PDFs in FAISS with metadata.")



def main():
    st.set_page_config("Chat PDF", layout="wide")
    st.header("Chat with PDF")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", 
                                    accept_multiple_files=True, type=["pdf"])
        
        if st.button("Submit & Process") and pdf_docs:
            with st.spinner("Processing..."):
                text_chunks = []  # To store each PDF as a separate chunk
                metadata = []  # To store filenames
                
                for pdf in pdf_docs:
                    pdf_texts = get_pdf_text(pdf_docs)  # Corrected
                    text_chunks, metadata = get_text_chunks(pdf_texts)  # Store full text as one chunk
                    # metadata.append({"filename": pdf.name})  # Store filename
                    # st.write("metadata:  ",metadata)
                get_vector_store(text_chunks, metadata)
                st.success("Processing Complete! PDFs are now searchable.")


    # Create a container for the chat history
    # chat_container = st.container()

    # # Display chat history
    # with chat_container:
    #     for message in st.session_state.chat_history:
    #         display_chat_message(message["role"], message["content"])

    # # Create the input box at the bottom with a submit button
    # col1, col2 = st.columns([6, 1])
    # with col1:
    #     st.text_input("Ask any Question from the PDF Files", 
    #                  key="user_input", 
    #                  on_change=handle_submit)
    
    # # Reset submitted flag when the input is empty
    # if not st.session_state.user_input:
    #     st.session_state.submitted = False

if __name__ == "__main__":
    main()