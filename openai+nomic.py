# import streamlit as st
# from PyPDF2 import PdfReader
# import os
# import pickle
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain.vectorstores import FAISS
# from langchain.docstore.document import Document
# from dotenv import load_dotenv
# from rank_bm25 import BM25Okapi
# from langchain_community.embeddings import HuggingFaceEmbeddings

# # Load OpenAI API Key
# load_dotenv()
# os.getenv("OPENAI_API_KEY")

# # Initialize session state for chat history
# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = []
# if 'user_input' not in st.session_state:
#     st.session_state.user_input = ''
# if 'submitted' not in st.session_state:
#     st.session_state.submitted = False

# # Global Variables for BM25
# bm25_corpus = []  # Tokenized chunks for BM25
# bm25_chunks = []  # Original chunks (for retrieval)
# bm25_index = None  # BM25 Model


# def get_pdf_text(pdf_docs):
#     """Extract text from PDFs and store metadata."""
#     documents = []
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         text = ""
#         for page in pdf_reader.pages:
#             extracted_text = page.extract_text()
#             if extracted_text:
#                 text += extracted_text + "\n"

#         doc = Document(page_content=text, metadata={"source": pdf.name})
#         documents.append(doc)

#         print(f"[INFO] Processed PDF: {pdf.name}")
#         st.sidebar.write(f"Processed: {pdf.name}")

#     return documents


# def get_text_chunks(documents):
#     """Split text into chunks and save them for BM25."""
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=250,
#         separators=["\n\n", "\n", ".", "!", ":"],
#     )

#     all_chunks = []
#     global bm25_corpus, bm25_chunks

#     with open("chunks.txt", "w", encoding="utf-8") as f:
#         for doc in documents:
#             chunks = text_splitter.create_documents(
#                 texts=[doc.page_content],
#                 metadatas=[doc.metadata]
#             )
#             all_chunks.extend(chunks)

#             for i, chunk in enumerate(chunks):
#                 f.write(f"Chunk {i+1} from {doc.metadata['source']}:\n{chunk.page_content}\n\n")
#                 bm25_corpus.append(chunk.page_content.split())  # Tokenize text for BM25
#                 bm25_chunks.append(chunk)

#             print(f"[INFO] Created {len(chunks)} chunks for {doc.metadata['source']}")
#             st.sidebar.write(f"Created {len(chunks)} chunks for {doc.metadata['source']}")

#     print("[INFO] Chunks saved to chunks.txt")
#     return all_chunks


# def get_vector_store(chunks):
#     """Create FAISS vector store and BM25 index."""
#     embeddings = HuggingFaceEmbeddings(
#         model_name="nomic-ai/nomic-embed-text-v1",
#         model_kwargs={'trust_remote_code': True}
#     )
#     vector_store = FAISS.from_documents(chunks, embedding=embeddings)

#     # Save FAISS index
#     vector_store.save_local("faiss_index")

#     # Create and Save BM25 index
#     global bm25_index
#     bm25_index = BM25Okapi(bm25_corpus)

#     with open("bm25_index.pkl", "wb") as f:
#         pickle.dump(bm25_index, f)

#     with open("bm25_chunks.pkl", "wb") as f:
#         pickle.dump(bm25_chunks, f)

#     print(f"[INFO] Created FAISS vector store with {len(chunks)} chunks")
#     print(f"[INFO] Created BM25 index with {len(bm25_chunks)} chunks")
#     st.sidebar.write(f"Created FAISS vector store with {len(chunks)} chunks")
#     st.sidebar.write(f"Created BM25 index with {len(bm25_chunks)} chunks")


# def format_docs(docs):
#     """Format retrieved documents for display."""
#     return "\n\n".join(f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}" for doc in docs)


# def bm25_search(query, k=5):
#     """Retrieve top-k results from BM25."""
#     global bm25_index, bm25_chunks

#     if bm25_index is None:
#         st.write("BM25 index is not available. Please upload and process your PDFs to create the index.")
#         return []

#     scores = bm25_index.get_scores(query.split())
#     ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

#     retrieved_docs = [bm25_chunks[i] for i in ranked_indices]
#     return retrieved_docs


# def hybrid_search(user_question, k=5):
#     """Perform Hybrid Retrieval (BM25 + FAISS)."""
#     embeddings = HuggingFaceEmbeddings(
#         model_name="nomic-ai/nomic-embed-text-v1",
#         model_kwargs={'trust_remote_code': True}
#     )
#     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

#     faiss_docs = new_db.similarity_search(user_question, k=k)
#     bm25_docs = bm25_search(user_question, k=k)

#     combined_docs = {doc.page_content: doc for doc in faiss_docs + bm25_docs}
#     return list(combined_docs.values())[:k]


# def user_input(user_question):
#     """Handle user queries and return answers."""
#     docs = hybrid_search(user_question, k=10)

#     print(f"\n[INFO] Retrieved {len(docs)} documents for query: {user_question}")
#     st.sidebar.write("Retrieved Documents:")
#     for i, doc in enumerate(docs):
#         print(f"[DEBUG] Document {i+1} (From: {doc.metadata.get('source', 'Unknown')})")
#         print(doc.page_content[:300])
#         st.sidebar.write(f"- From: {doc.metadata.get('source', 'Unknown')}")

#     sources = list(set(doc.metadata.get('source', 'Unknown') for doc in docs))
#     sources_str = ', '.join(sources)

#     messages = [
#         {
#         "role": "system",
#         "content": f"""You are an AI assistant that helps users understand PDF documents. 
#         The following information comes from these PDF files: {sources_str}
        
#         **IMPORTANT: Response Format**  
#         - If you find relevant information: **"Sources: [list of PDF filenames, comma-separated]"**  
#         - If you don't find relevant information: **"No relevant information found in the provided PDFs."**  
#         - After stating the sources, provide a detailed and structured answer.

#         **Rules for Answering:**  
#         - Use **only** the provided context to generate responses.   
#         - If the answer is **not available**, state: **"Answer is not available in the context."** Do not generate speculative or misleading answers.  
#         - If the query involves **calculations**, perform them and provide the exact result.    
#         - If the query is **unclear**, ask for clarification instead of making assumptions.  

#         **Conversation Memory:**  
#         - Support **multi-turn conversations** by remembering previous interactions.  
#         - Reference prior interactions when relevant to maintain consistency.  

#         Maintain an appropriate tone—**formal, conversational, concise, or elaborate**, depending on the query.  
#         """
#     },
#     {
#         "role": "user",
#         "content": f"""Question: {user_question}
        
#         Information from PDFs:
#         {format_docs(docs)}"""
#     }
#     ]

#     model = ChatOpenAI(model="gpt-4o", temperature=0.3)
#     response = model.invoke(messages)

#     print(f"\n[INFO] GPT-4o Response:\n{response.content}\n")
#     return response.content

# def display_chat_message(role, content):
#     with st.container():
#         if role == "user":
#             st.markdown(f"""
#             <div style="display: flex; justify-content: flex-end;">
#                 <div style="background-color: #007AFF; color: white; padding: 10px; 
#                 border-radius: 15px; margin: 5px; max-width: 70%;">
#                      {content}
#                 </div>
#             </div>
#             """, unsafe_allow_html=True)
#         else:
#             st.markdown(f"""
#             <div style="display: flex; justify-content: flex-start;">
#                 <div style="background-color: #E9ECEF; color: black; padding: 10px; 
#                 border-radius: 15px; margin: 5px; max-width: 70%;">
#                     🤖 {content}
#                 </div>
#             </div>
#             """, unsafe_allow_html=True)

# def handle_submit():
#     if st.session_state.user_input and not st.session_state.submitted:
#         user_question = st.session_state.user_input
        
#         # Add user message to chat history
#         st.session_state.chat_history.append({"role": "user", "content": user_question})
        
#         # Get AI response
#         response = user_input(user_question)
        
#         # Add AI response to chat history
#         st.session_state.chat_history.append({"role": "assistant", "content": response})
        
#         # Clear the input and set submitted flag
#         st.session_state.user_input = ''
#         st.session_state.submitted = True


# def main():
#     """Main Streamlit UI."""
#     st.set_page_config("Chat PDF", layout="wide")
#     st.header("Chat with PDF(nomic and gpt4o)")

#     with st.sidebar:
#         st.title("Menu:")
#         pdf_docs = st.file_uploader("Upload your PDFs", accept_multiple_files=True)
#         if st.button("Submit & Process"):
#             with st.spinner("Processing..."):
#                 if pdf_docs:
#                     documents = get_pdf_text(pdf_docs)
#                     text_chunks = get_text_chunks(documents)
#                     get_vector_store(text_chunks)
#                     st.success("Processing complete! Chunks saved to chunks.txt")
#                 else:
#                     st.error("Please upload PDFs first.")

#     # st.text_input("Ask any Question from PDFs", key="user_input", on_change=lambda: user_input(st.session_state.user_input))
#     # Create a container for the chat history
#     chat_container = st.container()

#     # Display chat history
#     with chat_container:
#         for message in st.session_state.chat_history:
#             display_chat_message(message["role"], message["content"])

#     # Create the input box at the bottom with a submit button
#     col1, col2 = st.columns([6, 1])
#     with col1:
#         st.text_input("Ask any Question from the PDF Files", 
#                      key="user_input", 
#                      on_change=handle_submit)
    
#     # Reset submitted flag when the input is empty
#     if not st.session_state.user_input:
#         st.session_state.submitted = False


# if __name__ == "__main__":
#     main()



import streamlit as st
from PyPDF2 import PdfReader
import os
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
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