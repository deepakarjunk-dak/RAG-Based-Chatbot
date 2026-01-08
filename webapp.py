import os
import google.generativeai as genai
# from langchain.vectorstores import FAISS  # This will be the vector database
from langchain_community.vectorstores import FAISS

from langchain_community.embeddings import HuggingFaceEmbeddings # To perform word embeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter # This for chunking
# from langchain_community_text_splitter import RecursiveCharacterTextSplitter
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.text_splitter import RecursiveCharacterTextSplitter

from langchain_text_splitters import RecursiveCharacterTextSplitter

from pypdf import PdfReader
import faiss
import streamlit as st
from pdfextractor import text_extractor_pdf

from dotenv import load_dotenv
load_dotenv()

# Create the main page
st.title(':green[RAG Based CHATBOT]')
tips = '''Follow the steps to use this application:
* Upload your pdf document in sidebar.
* Write your query and start chatting with the bot.'''
st.subheader(tips)

# Load PDF in Side Bar
st.sidebar.title(':orange[UPLOAD YOUR DOCUMENT HERE (PDF Only)]')
file_uploaded = st.sidebar.file_uploader('Upload File')
if file_uploaded:
    file_text = text_extractor_pdf(file_uploaded)
    # Step 1: Configure the models

    # Configure LLM
    key = os.getenv('GOOGLE_API_KEY')
    genai.configure(api_key=key)
    llm_model = genai.GenerativeModel('gemini-2.5-flash-lite')

    # Configure Embedding Model
    embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

    # Step 2 : Chunking (Create Chunks)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap = 200)
    chunks = splitter.split_text(file_text)

    # Step 3: Create FAISS Vector Store
    vector_store = FAISS.from_texts(chunks,embedding_model)

    # Step 4: Configure retriever
    retriever = vector_store.as_retriever(search_kwargs={"k":3})

    # Lets create a function that takes query and return the generated text
    def generate_response(query):
        # Step 6 : Retrieval (R)
        # retrived_docs = retriever.get_relevant_documents(query=query)
        retrived_docs = retriever.invoke(query)
        context = ' '.join([doc.page_content for doc in retrived_docs])

        # Step 7: Write a Augmeneted prompt (A)
        prompt = f'''You are a helpful assitant using RAG
        Here is the context = {context}
        The query asked by user is as follows = {query}'''

        # Step 9: Generation (G)
        content = llm_model.generate_content(prompt)
        return content.text

    
    # Lets create a chatbot in order to start the converstaion
    # Initialize chat if there is no history
    if 'history' not in st.session_state:
        st.session_state.history = []

    # Display the History
    for msg in st.session_state.history:
        if msg['role'] == 'user':
            # st.write(f':green[User:] :blue[{msg['text']}]')
            st.write(f":green[User:] :blue[{msg['text']}]")

        else:
            # st.write(f':orange[Chatbot:]  {msg['text']}')
            st.write(f":orange[Chatbot:] {msg['text']}")


    # Input from the user (Using Steamlit Form)
    with st.form('Chat Form',clear_on_submit=True):
        user_input = st.text_input('Enter Your Text Here: ')
        send = st.form_submit_button('Send')
    
    # Start the converstaion and append the output and query in history
    if user_input and send:

        st.session_state.history.append({"role":'user',"text":user_input})

        model_output = generate_response(user_input)

        st.session_state.history.append({'role':'chatbot','text':model_output})

        st.rerun()