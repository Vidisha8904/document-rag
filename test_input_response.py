# test input and response foramt of embedding.


import streamlit as st
from PyPDF2 import PdfReader
import os
import pickle
import numpy as np
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
        chunk_size=2000,
        chunk_overlap=200,
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

# new
def display_faiss_contents():
    from langchain.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings

    # Load the FAISS index
    embeddings = HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1",
        model_kwargs={'trust_remote_code': True}
    )
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    # Retrieve all stored documents and display them
    documents = vector_store.docstore._dict.values()
    debug_text = ""
    for i, doc in enumerate(documents):
        debug_text += f"\nDocument {i+1}:\n"
        debug_text += f"Metadata: {doc.metadata}\n"
        debug_text += f"Content:\n{doc.page_content[:500]}\n"  # Show first 500 characters

    num_vectors = len(vector_store.docstore._dict)
    embeddings_array = vector_store.index.reconstruct_n(0, num_vectors)
    debug_text += f"\nEmbeddings shape: {embeddings_array.shape}\n"
    if num_vectors > 0:
        debug_text += f"Sample embedding (first vector): {np.array2string(embeddings_array[0], precision=3)}\n"

    # Print to console (or use st.write to show in the app)
    print(debug_text)
    st.text_area("FAISS Index Contents", debug_text, height=300)


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
# Add a debug button to view FAISS index contents
    if st.button("View FAISS Index Contents"):
        display_faiss_contents()
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















# import streamlit as st
# from PyPDF2 import PdfReader
# import os
# import pickle
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain.vectorstores import FAISS
# from langchain.docstore.document import Document
# from dotenv import load_dotenv
# from langchain_community.embeddings import HuggingFaceEmbeddings

# load_dotenv()
# os.getenv("OPENAI_API_KEY")

# # Initialize session state for chat history and submitted flag
# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = []
# if 'user_input' not in st.session_state:
#     st.session_state.user_input = ''
# if 'submitted' not in st.session_state:
#     st.session_state.submitted = False

# def get_pdf_text(pdf_docs):
#     documents = []
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         text = ""
#         for page_num, page in enumerate(pdf_reader.pages):
#             text += page.extract_text() + "\n"
        
#         # Store complete text with metadata
#         doc = Document(
#             page_content=text,
#             metadata={"source": pdf.name}
#         )
#         documents.append(doc)
        
#         # Debug print
#         st.sidebar.write(f"Processed: {pdf.name}")
#     return documents

# def get_text_chunks(documents):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=100,
#     )
    
#     all_chunks = []
#     for doc in documents:
#         chunks = text_splitter.create_documents(
#             texts=[doc.page_content],
#             metadatas=[doc.metadata]
#         )
#         all_chunks.extend(chunks)
        
#         # Debug print
#         st.sidebar.write(f"Created {len(chunks)} chunks for {doc.metadata['source']}")
    
#     return all_chunks

# def get_vector_store(chunks):
#     embeddings = HuggingFaceEmbeddings(
#         model_name="nomic-ai/nomic-embed-text-v1",
#         model_kwargs={'trust_remote_code': True}
#     )
#     vector_store = FAISS.from_documents(chunks, embedding=embeddings)
    
#     # Debug print
#     st.sidebar.write(f"Created vector store with {len(chunks)} chunks")
    
#     vector_store.save_local("faiss_index")

# def format_docs(docs):
#     return "\n\n".join(f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}" for doc in docs)

# def user_input(user_question):
#     embeddings = HuggingFaceEmbeddings(
#         model_name="nomic-ai/nomic-embed-text-v1",
#         model_kwargs={'trust_remote_code': True}
#     )
#     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
#     # Get relevant documents
#     docs = new_db.similarity_search(user_question, k=4)
    
#     # Debug print
#     st.sidebar.write("Retrieved Documents:")
#     for doc in docs:
#         st.sidebar.write(f"- From: {doc.metadata.get('source', 'Unknown')}")
    
#     # Extract unique sources
#     sources = list(set(doc.metadata.get('source', 'Unknown') for doc in docs))
#     sources_str = ', '.join(sources)
    
#     # Create a custom system message that forces mention of sources
#     messages = [
#     {
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

    
#     # Use ChatOpenAI directly for more control
#     model = ChatOpenAI(model="gpt-4o", temperature=0.3)
#     response = model.invoke(messages)
    
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
#     st.set_page_config("Chat PDF", layout="wide")
#     st.header("Chat with PDF - nomic and gpt")

#     with st.sidebar:
#         st.title("Menu:")
#         pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", 
#                                   accept_multiple_files=True)
#         if st.button("Submit & Process"):
#             with st.spinner("Processing..."):
#                 if pdf_docs:
#                     documents = get_pdf_text(pdf_docs)
#                     text_chunks = get_text_chunks(documents)
#                     # st.write(text_chunks)
                    
#                     # Save all the chunks in one object named 'input_obj'
#                     input_obj = {"chunks": text_chunks}
#                     # Preview the first 2 chunks from input_obj
#                     st.write("Input Object Preview:", input_obj["chunks"][:2])
                    
#                     # Build and save the FAISS vector store
#                     get_vector_store(text_chunks)
                    
#                     # Load the saved vector store to create the 'response_obj'
#                     embeddings = HuggingFaceEmbeddings(
#                         model_name="nomic-ai/nomic-embed-text-v1",
#                         model_kwargs={'trust_remote_code': True}
#                     )
#                     vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
                    
#                     # Build the response object by extracting all documents with embeddings
#                     response_obj = {"documents": []}
#                     # Preview the first 2 documents from response_obj
#                     # st.write("Response Object Preview:", response_obj["documents"][:6])


#                     documents_list = list(vector_store.docstore._dict.values())
#                     for i, doc in enumerate(documents_list):
#                         embedding = vector_store.index.reconstruct(i)
#                         response_obj["documents"].append({
#                             "metadata": doc.metadata,
#                             "content": doc.page_content,
#                             "embedding": embedding.tolist()  # converting numpy array to list if needed
#                         })
                    
#                     # (Optional) You can view the objects for debugging:
#                     st.write("Input Object:", input_obj)
#                     st.write("Response Object:", response_obj)
                    
#                     st.success("Done")
#                 else:
#                     st.error("Please upload PDF files first.")
    
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
