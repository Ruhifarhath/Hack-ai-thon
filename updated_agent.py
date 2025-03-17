import os
import streamlit as st
import fitz  # PyMuPDF for extracting text from PDF
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from crewai import Agent, Task, Crew  

# 🔧 Disable TensorFlow oneDNN Warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# 🎨 Streamlit UI Setup
st.set_page_config(page_title="AI PDF Q&A", layout="wide")
st.title("📄 AI-Powered PDF Knowledge System")

# 📜 Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = "\n".join(page.get_text("text") for page in doc)
    return text

# 🧠 Load DeepSeek-R1 Model from Hugging Face
@st.cache_resource  # Cache to avoid reloading
def load_model():
    model_name = "deepseek-ai/DeepSeek-R1"  # ✅ Hugging Face DeepSeek Model

    # Detect GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32  # ✅ Avoid FP8 issues (float16 for GPU, float32 for CPU)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,  # ✅ Use float16 for GPU, float32 for CPU
        device_map="auto" if torch.cuda.is_available() else "cpu",  # ✅ Force CPU if no CUDA available
        trust_remote_code=True
    )

    return tokenizer, model

# 🚀 Load Model
try:
    tokenizer, model = load_model()
    st.success("✅ DeepSeek-R1 Model loaded successfully from Hugging Face!")
except Exception as e:
    st.error(f"❌ Model loading failed: {str(e)}")

# 📝 Function to generate AI responses
def generate_response(prompt):
    if "tokenizer" not in globals() or "model" not in globals():
        return "⚠️ Model failed to load. Please check error messages."

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_length=512)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# 📂 PDF Upload UI
uploaded_file = st.file_uploader("📤 Upload a PDF document", type="pdf")

if uploaded_file:
    pdf_text = extract_text_from_pdf(uploaded_file)
    st.success("✅ PDF successfully uploaded and processed!")

    # 🏗️ Define AI Agents
    expert_extraction_agent = Agent(
        role="Knowledge Extractor",
        goal="Extract key insights from documents.",
        backstory="AI assistant designed for extracting relevant information.",
        verbose=False
    )

    retrieval_agent = Agent(
        role="Query Handler",
        goal="Retrieve the most relevant information based on user queries.",
        backstory="AI trained in document retrieval and NLP.",
        verbose=False
    )

    # 📌 Define Tasks
    expert_extraction_task = Task(
        description="Extract important insights from the uploaded PDF.",
        agent=expert_extraction_agent,
        expected_output="Summarized insights from the document.",
    )

    retrieval_task = Task(
        description="Retrieve knowledge efficiently based on user queries.",
        agent=retrieval_agent,
        expected_output="Relevant and structured responses to user queries.",
        context=[expert_extraction_task]
    )

    # 🎯 Crew Setup
    crew = Crew(
        agents=[expert_extraction_agent, retrieval_agent],
        tasks=[expert_extraction_task, retrieval_task]
    )

    # 🔄 Run Crew AI Workflow
    st.info("⚙️ Processing the document with AI...")
    try:
        crew.kickoff()
        st.success("✅ Document processing completed! You can now ask questions.")
    except Exception as e:
        st.error(f"❌ Error occurred: {e}")

    # 🔍 Question-Answering UI
    user_query = st.text_input("🔍 Ask a question about the document:")

    if user_query:
        with st.spinner("🤖 Thinking..."):
            try:
                response = generate_response(f"Context: {pdf_text}\nQuestion: {user_query}\nAnswer:")
                st.write("🧠 **AI Answer:**", response)
            except Exception as e:
                st.error(f"❌ Failed to retrieve an answer: {e}")
