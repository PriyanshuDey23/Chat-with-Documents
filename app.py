import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from prompt import *
from docx import Document

# Load environment variables
load_dotenv()

# Configure API key for Google Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize embeddings model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Streamlit page configuration
st.set_page_config(page_title="Interact with PDFs, DOCX, and TXT Files", layout="wide")

# Ensure persistent storage for uploaded files
UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Function to extract text from a PDF
def extract_pdf_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

# Function to extract text from a DOCX file
def extract_docx_text(docx_file):
    doc = Document(docx_file)
    return "\n".join([para.text for para in doc.paragraphs])

# Function to extract text from a TXT file
def extract_txt_text(txt_file):
    return txt_file.read().decode("utf-8")

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_text(text)

# Function to store text chunks in vector database
def get_vector_store(chunks):
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    vectorstore.save_local("faiss_index")

# Function to create a QA chain
def get_conversational_chain():
    prompt_template = PROMPT  # Imported prompt
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-8b", temperature=0.2)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "user_question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Function to process user input
def user_input(user_question):
    if not os.path.exists("faiss_index"):
        st.error("No processed documents found. Please upload and process documents first.")
        return

    new_db = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "user_question": user_question}, return_only_outputs=True)
    
    # Display the question and answer in chat format
    with st.chat_message("user"):
        st.markdown(f"**You:** {user_question}")

    with st.chat_message("assistant"):
        st.markdown(f"**Bot:** {response['output_text']}")

# Main function for the Streamlit app
def main():
    st.title("📄 Chat with Documents")

    # Sidebar for file upload
    with st.sidebar:
        st.header("📂 Upload Documents")
        uploaded_files = st.file_uploader("Upload documents (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)
        
        if st.button("Process Documents"):
            if uploaded_files:
                with st.spinner("Processing documents..."):
                    all_chunks = []
                    
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.read())

                        if uploaded_file.name.endswith(".pdf"):
                            raw_text = extract_pdf_text(file_path)
                        elif uploaded_file.name.endswith(".docx"):
                            raw_text = extract_docx_text(file_path)
                        elif uploaded_file.name.endswith(".txt"):
                            raw_text = extract_txt_text(uploaded_file)
                        else:
                            continue

                        if raw_text.strip():
                            text_chunks = get_text_chunks(raw_text)
                            all_chunks.extend(text_chunks)
                        else:
                            st.warning(f"⚠️ {uploaded_file.name} contains no extractable text.")
                    
                    if all_chunks:
                        get_vector_store(all_chunks)
                        st.success("✅ Documents processed successfully!")
                    else:
                        st.warning("❌ No valid text found in documents.")

    # Main chat interface
    st.subheader("💬 Ask a Question")
    user_question = st.text_input("Enter your question and press Enter:", key="user_input")
    
    if st.button("Submit Question"):
        if user_question:
            user_input(user_question)
        
            

if __name__ == "__main__":
    main()
