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
import pdfplumber
import json
import traceback

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
        # Save to temporary file for camelot (since it needs a file path)
        temp_path = f"temp_{pdf.name}"
        with open(temp_path, "wb") as f:
            f.write(pdf.read())
        
        # Extract text with PyPDF2 (or pdfplumber for better layout)
        text_data = {"text": []}
        try:
            pdf_reader = PdfReader(temp_path)
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text().strip()
                if page_text:  # Only add non-empty text
                    text_data["text"].append({"page": page_num + 1, "content": page_text})
        except Exception as e:
            st.sidebar.write(f"Error extracting text from {pdf.name}: {str(e)}")
        
        # Extract tables with camelot
        table_data = {"tables": []}
        table_count = 0
        try:
            # Try 'lattice' flavor first (for bordered tables)
            tables = camelot.read_pdf(temp_path, pages='all', flavor='lattice')
            for i, table in enumerate(tables):
                if table.df.size > 0:  # Check if table is non-empty
                    table_dict = {
                        "table_id": f"Table {i + 1}",
                        "page": table.parsing_report['page'],
                        "data": table.df.replace({None: ""}).to_dict(orient='records')  # Convert to list of dicts
                    }
                    table_data["tables"].append(table_dict)
                    table_count += 1
            
            # If no tables found, try 'stream' flavor (for borderless tables)
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
            st.sidebar.write(f"Exception type: {type(e).__name__}")
            st.sidebar.write(f"Traceback: {traceback.format_exc()}")
        
        # Combine text and tables into a single JSON object
        combined_data = {
            "source": pdf.name,
            "text": text_data["text"],
            "tables": table_data["tables"]
        }
        json_content = json.dumps(combined_data, ensure_ascii=False, indent=2)
        print(f"\nJSON Content for {pdf.name}:")
        # print(json_content)
        
        doc = Document(
            page_content=json_content,  # Store as JSON string
            metadata={"source": pdf.name}
        )
        documents.append(doc)
        
        st.sidebar.write(f"Processed: {pdf.name} - Found {table_count} tables and text extracted")

        # with open("output.json", "w", encoding="utf-8") as json_file:
        #     json.dump(combined_data, json_file, ensure_ascii=False, indent=2)
        
        # Clean up temporary file
        os.remove(temp_path)
    
    return documents

def get_text_chunks(documents):
    # Splitter for text-only content
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=250,
        separators=["\n\n", "\n", ","]  # Natural text boundaries
    )
    
    all_chunks = []
    for doc in documents:
        # Parse JSON content from the document
        data = json.loads(doc.page_content)
        
        # Extract text and tables
        text_entries = data.get("text", [])  # List of {"page": N, "content": "..."}
        table_entries = data.get("tables", [])  # List of {"table_id": "...", "page": N, "data": [...]}
        
        # Process text entries into chunks
        text_content = "\n".join(entry.get("content", "") for entry in text_entries if entry.get("content"))
        if text_content:
            text_chunks = text_splitter.create_documents(
                texts=[text_content],
                metadatas=[{"source": doc.metadata["source"], "type": "text"}]
            )
            all_chunks.extend(text_chunks)
        
        # Process each table as a single chunk
        for table in table_entries:
            # Convert table to a standalone JSON string
            table_json = json.dumps(table, ensure_ascii=False)
            # Create a Document object with the table JSON as content
            table_chunk = Document(
                page_content=table_json,
                metadata={"source": doc.metadata["source"], "type": "table"}
            )
            all_chunks.append(table_chunk)
        
        # Report chunk count
        total_chunks = len([c for c in all_chunks if c.metadata["source"] == doc.metadata["source"]])
        text_chunk_count = len([c for c in all_chunks if c.metadata["source"] == doc.metadata["source"] and c.metadata["type"] == "text"])
        table_chunk_count = len(table_entries)
        st.sidebar.write(f"Created {total_chunks} chunks for {doc.metadata['source']} "
                        f"({text_chunk_count} text, {table_chunk_count} tables)")
    
    # # --- Start of chunk saving to txt file (remove this block to disable) ---
    # with open("chunks_output.txt", "w", encoding="utf-8") as txt_file:
    #     for i, chunk in enumerate(all_chunks, 1):
    #         txt_file.write(f"Chunk {i}:\n")
    #         txt_file.write(f"Source: {chunk.metadata.get('source', 'Unknown')}\n")
    #         txt_file.write(f"Type: {chunk.metadata.get('type', 'unknown')}\n")
    #         txt_file.write(f"Content: {chunk.page_content}\n")
    #         txt_file.write("-" * 50 + "\n")
    # # --- End of chunk saving to txt file ---
    
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
    
    docs = new_db.similarity_search(user_question, k=6)
    
    st.sidebar.write("Retrieved Documents:")

    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get('source', 'Unknown')
        content_snippet = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
        
        # # Sidebar output (visible in Streamlit app)
        # st.sidebar.write(f"Document {i}:")
        # st.sidebar.write(f"- From: {source}")
        # st.sidebar.write(f"- Content snippet: {content_snippet}")
        # st.sidebar.write("---")  # Separator for readability
        
        # Console output (for debugging in terminal)
        print(f"\nDocument {i}:")
        print(f"Source: {source}")
        print(f"Full Content: {doc.page_content}")
        print("---")
    for doc in docs:
        st.sidebar.write(f"- From: {doc.metadata.get('source', 'Unknown')}")
    
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

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

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