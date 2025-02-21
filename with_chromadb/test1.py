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
        pdf_reader = PdfReader(pdf)
        text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            text += page.extract_text() + "\n"
        
        # Store complete text with metadata
        doc = Document(
            page_content=text,
            metadata={"source": pdf.name}
        )
        documents.append(doc)
        
        # Debug print
        st.sidebar.write(f"Processed: {pdf.name}")
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
        
        # Extract text and add to ChromaDB with source information
        chunk_texts = [chunk.page_content for chunk in chunks]
        add_to_collection(chunk_texts, doc.metadata['source'])
        
        # Debug print
        st.sidebar.write(f"Created {len(chunks)} chunks for {doc.metadata['source']}")
    
    return all_chunks

def format_docs(docs, sources):
    formatted_docs = []
    for doc_text, source in zip(docs, sources):
        formatted_docs.append(f"Source: {source}\n{doc_text}")
    return "\n\n".join(formatted_docs)

def user_input(user_question):
    # Get relevant documents using ChromaDB
    docs = retrieve_from_collection(user_question, top_k=4)
    
    # Extract sources from the metadata (assuming retrieve_from_collection returns metadata)
    sources = [doc['metadata']['filename'] for doc in docs] if isinstance(docs, list) and docs and 'metadata' in docs[0] else ['Unknown'] * len(docs)
    doc_texts = [doc['document'] for doc in docs] if isinstance(docs, list) and docs and 'document' in docs[0] else docs
    
    
    # Debug print
    st.sidebar.write("Retrieved Documents:")
    for source in set(sources):
        st.sidebar.write(f"- From: {source}")
    
    # Extract unique sources
    unique_sources = list(set(sources))
    sources_str = ', '.join(unique_sources)
    
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
        {format_docs(doc_texts, sources)}"""
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
    st.header("Chat with PDF - ChromaDB and GPT")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", 
                                  accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                if pdf_docs:
                    documents = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(documents)
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