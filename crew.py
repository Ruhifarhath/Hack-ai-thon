import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure dependencies are installed
# Run: pip install transformers torch streamlit accelerate

@st.cache_resource
def load_model():
    model_name = "deepseek-ai/DeepSeek-R1"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load model with proper configurations
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,  # Handles GPU/CPU
        device_map="auto" if torch.cuda.is_available() else None,  # Ensures correct device usage
        low_cpu_mem_usage=True  # Optimizes memory on CPU
    )

    return tokenizer, model

# Streamlit UI
st.set_page_config(page_title="DeepSeek AI Model", layout="wide")
st.title("🔍 AI-Powered PDF Knowledge System")

# Load Model
st.info("⏳ Loading model, please wait...")
try:
    tokenizer, model = load_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Model loading failed: {str(e)}")

# User Input
user_input = st.text_area("Ask a question:")
if st.button("Generate Answer"):
    if user_input:
        inputs = tokenizer(user_input, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_length=200)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        st.write("🧠 **AI Response:**", response)
    else:
        st.warning("⚠️ Please enter a question.")
