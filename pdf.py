import re
import streamlit as st
from huggingface_hub import login
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from PIL import Image
import pdf2image
from transformers import AutoTokenizer, AutoModelForDocumentQuestionAnswering, pipeline

# 🔹 Authenticate with Hugging Face
HUGGINGFACE_TOKEN = ""  # 🔹 Replace with your actual token
login(HUGGINGFACE_TOKEN)

st.markdown("""
    <style>
    .stApp { background-color: #00000; color: #ffffff; }
    .stChatInput input { background-color: #1E1E1E !important; color: #0000 !important; border: 1px solid #3A3A3A !important; }
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) { background-color: #1E1E1E !important; border: 1px solid #3A3A3A !important; color: #E0E0E0 !important; border-radius: 10px; padding: 15px; margin: 10px 0; }
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) { background-color: #2A2A2A !important; border: 1px solid #404040 !important; color: #F0F0F0 !important; border-radius: 10px; padding: 15px; margin: 10px 0; }
    .stFileUploader { background-color: #1E1E1E; border: 1px solid #3A3A3A; border-radius: 5px; padding: 15px; }
    h1, h2, h3 { color: #00FFAA !important; }
    </style>
    """, unsafe_allow_html=True)

PDF_STORAGE_PATH = 'document_store/pdfs/'
EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
MODEL_NAME = "impira/layoutlm-document-qa"

# 🔹 Load the LayoutLM model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HUGGINGFACE_TOKEN)
model = AutoModelForDocumentQuestionAnswering.from_pretrained(MODEL_NAME, token=HUGGINGFACE_TOKEN)

# 🔹 Create a document QA pipeline
qa_pipeline = pipeline("document-question-answering", model=model, tokenizer=tokenizer)

# 🔹 Function to remove <think> tags (not needed for LayoutLM, but kept for consistency)
def remove_think_tags(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

def save_uploaded_file(uploaded_file):
    file_path = PDF_STORAGE_PATH + uploaded_file.name
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

def load_pdf_documents(file_path):
    document_loader = PDFPlumberLoader(file_path)
    return document_loader.load()

def chunk_documents(raw_documents):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_processor.split_documents(raw_documents)

def index_documents(document_chunks):
    DOCUMENT_VECTOR_DB.add_documents(document_chunks)

def pdf_to_images(pdf_path):
    """ Converts a PDF into a list of images (one per page). """
    images = pdf2image.convert_from_path(pdf_path)
    return images

def generate_answer(user_query, pdf_images):
    """ Uses LayoutLM to answer questions based on document images. """
    if not pdf_images:
        return "I couldn't extract any images from the document."

    # 🔹 Pass the first page as an image (or modify to process multiple pages)
    response = qa_pipeline(image=pdf_images[0], question=user_query)

    if response:
        answer = response[0]['answer']
        return remove_think_tags(answer)
    
    return "I'm not sure about the answer."

# 🔹 UI Configuration
st.title("📘 DocuMind AI")
st.markdown("### Your Intelligent Document Assistant")
st.markdown("---")

# 🔹 File Upload Section
uploaded_pdf = st.file_uploader(
    "Upload Research Document (PDF)",
    type="pdf",
    help="Select a PDF document for analysis",
    accept_multiple_files=False
)

if uploaded_pdf:
    saved_path = save_uploaded_file(uploaded_pdf)
    pdf_images = pdf_to_images(saved_path)  # Convert PDF to images
    
    st.success("✅ Document processed successfully! Ask your questions below.")
    
    user_input = st.chat_input("Enter your question about the document...")
    
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        
        with st.spinner("Analyzing document..."):
            ai_response = generate_answer(user_input, pdf_images)
            
        with st.chat_message("assistant", avatar="🤖"):
            st.write(ai_response)
