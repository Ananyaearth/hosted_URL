import streamlit as st
import pandas as pd
import numpy as np
import faiss
import os
import pickle
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Set page config
st.set_page_config(page_title="SHL Assessment Recommender", layout="wide")

# Load Gemini API key from Streamlit secrets
genai.configure(api_key="AIzaSyASKTzSNuMbJMdZWr81Xuw2hS1Poe3acZo")

# Title
st.title("🔍 SHL Assessment Recommender")

# Load data and model
@st.cache_resource
def load_data():
    df = pd.read_csv("shl_catalog_detailed.csv")

    # Prepare structured documents
    documents = []
    for i, row in df.iterrows():
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
        documents.append(content.strip())

    return documents

@st.cache_resource
def load_model_and_index():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index = faiss.read_index("index.faiss")  # Precomputed index file
    with open("index_mapping.pkl", "rb") as f:
        index_to_doc = pickle.load(f)
    return model, index, index_to_doc

documents = load_data()
model, faiss_index, index_to_doc = load_model_and_index()

# Search function
def search_documents(query, top_k=10):
    query_embedding = model.encode([query]).astype('float32')
    distances, indices = faiss_index.search(query_embedding, top_k)
    return [index_to_doc[i] for i in indices[0]]

# Gemini RAG
def ask_rag_question(query):
    context_chunks = search_documents(query)
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
    response = genai.generate_content(prompt)
    return response.text

# Streamlit UI
query = st.text_input("🔎 Enter your hiring requirement (e.g., Python developer with collaboration skills)...")

if query:
    with st.spinner("Thinking... 🤔"):
        answer = ask_rag_question(query)
        st.markdown(answer, unsafe_allow_html=True)
