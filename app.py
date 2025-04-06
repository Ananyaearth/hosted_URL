import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os

# Streamlit config
st.set_page_config(page_title="SHL Assessment Recommender", layout="wide")
st.title("🔍 SHL Assessment Recommender")

# Gemini API key from env (use GitHub Secrets for deployment)
genai.configure(api_key=os.getenv("GEMINI_API_KEY", "AIzaSyASKTzSNuMbJMdZWr81Xuw2hS1Poe3acZo"))

# Load dataset and FAISS
def load_resources():
    try:
        df = pd.read_csv("shl_catalog_detailed.csv")
        faiss_index = faiss.read_index("shl_catalog_index.faiss")
        return df, faiss_index
    except Exception as e:
        st.error(f"Failed to load data or index: {e}")
        st.stop()

@st.cache_resource
def load_model():
    try:
        # Load the new model from the bundled folder
        model_path = os.path.join(os.path.dirname(__file__), "models", "multi-qa-MiniLM-L6-cos-v1")
        model = SentenceTransformer(model_path, device='cpu')
        return model
    except Exception as e:
        st.error(f"Failed to load embedding model: {e}")
        st.info("Ensure the 'models/multi-qa-MiniLM-L6-cos-v1' folder contains the model files (config.json, pytorch_model.bin, etc.).")
        st.stop()

def get_document_from_index(df, i):
    row = df.iloc[i]
    content = f"""
    Assessment Name: {row['Individual Test Solutions']}
    URL: {row['URL']}
    Description: {row['Description']}

    Remote Testing Support: {row['Remote Testing (y/n)']}
    Adaptive/IRT Support: {row['Adaptive/IRT (y/n)']}
    Assessment Length: {row['Assessment Length']}
    Test Type: {row['Test Type']}

    Job Levels: {row['Job Levels']}
    Languages: {row['Languages']}
    """
    return content.strip()

def search_documents(query, model, faiss_index, df, top_k=10):
    query_embedding = model.encode([query]).astype('float32')
    distances, indices = faiss_index.search(query_embedding, top_k)
    return [get_document_from_index(df, i) for i in indices[0] if i != -1]

def ask_rag_question(query, model, faiss_index, df):
    context_chunks = search_documents(query, model, faiss_index, df)
    context = "\n\n".join(context_chunks)

    prompt = f"""
You are an expert in HR assessments. Based on the context below, identify all assessments relevant to the user's request.

### Instructions:
- Present your answer in a markdown table with **these columns**:
  | Assessment Name (with link) | Remote Testing Support | Adaptive/IRT Support | Duration & Test Type | Why Recommended/Not Recommended |
- Carefully review **all the provided context chunks** and extract multiple assessments if applicable.
- **Do not make up any data** — only use what's in the context.
- **Do not hallucinate or assume** Remote Testing or Adaptive/IRT support if it is not explicitly mentioned.

### Context:
{context}

### Question:
{query}

### Answer:
"""
    try:
        response = genai.generate_content(prompt)
        return response.text or "⚠️ Gemini returned an empty response."
    except Exception as e:
        return f"⚠️ Gemini failed: {e}"

# Load everything
df, faiss_index = load_resources()
model = load_model()

# User input
query = st.text_input("🔎 Enter your hiring requirement (e.g., Python developer with collaboration skills)...")

# Output
if query:
    with st.spinner("Thinking... 🤔"):
        answer = ask_rag_question(query, model, faiss_index, df)
        st.markdown(answer, unsafe_allow_html=True)
