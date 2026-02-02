import streamlit as st
import requests
from sentence_transformers import SentenceTransformer
import msgpack
import json
import os
from utils import extract_text_from_pdf, chunk_text, generate_answer_mock

# --- Page Config & Styling ---
st.set_page_config(page_title="Endee Agentic RAG", layout="wide", page_icon="🧬")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stHeader { color: #4F8BF9; }
    .agent-thought {
        font-style: italic;
        color: #FFD700;
        border-left: 2px solid #4F8BF9;
        padding-left: 10px;
        margin: 10px 0;
    }
    .result-card {
        padding: 15px;
        border-radius: 8px;
        background-color: #262730;
        margin-bottom: 10px;
        border-left: 4px solid #4F8BF9;
        color: #FFFFFF;
    }
    .result-card b { color: #4F8BF9; }
    .stInfo { background-color: #1e2130; border: 1px solid #4F8BF9; color: #FFFFFF; }
    </style>
    """, unsafe_allow_html=True)

# --- Session State to store raw text chunks ---
if "chunks_store" not in st.session_state:
    st.session_state.chunks_store = {}

# --- Initialization ---
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()
ENDEE_URL = "http://localhost:8080/api/v1"
INDEX_NAME = "agentic_knowledge_base"

# --- Sidebar: Admin & Ingestion ---
st.sidebar.title("🧬 Endee Agentic Panel")

if st.sidebar.button("Initialize Master Index"):
    with st.spinner("Creating Index..."):
        res = requests.post(f"{ENDEE_URL}/index/create", json={
            "index_name": INDEX_NAME,
            "dim": 384,
            "space_type": "cosine"
        })
        if res.status_code == 200:
            st.sidebar.success("Index Ready!")
        else:
            st.sidebar.warning(f"Note: {res.text}")

st.sidebar.header("📥 Data Ingestion")
uploaded_file = st.sidebar.file_uploader("Upload research PDFs", type=["pdf", "txt"])

if uploaded_file and st.sidebar.button("Knowledge Ingest"):
    with st.spinner("Processing Document..."):
        if uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file.read())
        else:
            text = uploaded_file.read().decode()
        
        chunks = chunk_text(text)
        st.sidebar.info(f"Split into {len(chunks)} chunks.")
        
        progress = st.sidebar.progress(0)
        success_count = 0
        for i, chunk in enumerate(chunks):
            doc_id = f"{uploaded_file.name}_chunk_{i}"
            # Store raw text in session state
            st.session_state.chunks_store[doc_id] = chunk
            
            res = requests.post(f"{ENDEE_URL}/index/{INDEX_NAME}/vector/insert", json={
                "id": doc_id,
                "vector": model.encode(chunk).tolist()
            })
            if res.status_code == 200:
                success_count += 1
            progress.progress((i + 1) / len(chunks))
        
        st.sidebar.success(f"Ingested {success_count} chunks!")

# --- Main UI ---
st.title("🤖 Agentic Research Assistant")
st.markdown("This assistant uses **Endee** for semantic retrieval and context awareness.")

query = st.text_input("Ask a question:", placeholder="e.g. explain me about this project")

if query:
    # Proactive Agent Logic: Always search DB unless it's a simple greeting
    greetings = ["hi", "hello", "hey", "hola", "namaste"]
    is_greeting = any(query.lower().strip() == g for g in greetings)
    
    if not is_greeting:
        st.markdown(f'<div class="agent-thought">Agent Action: Query is knowledge-based. Searching Endee Vector DB for context...</div>', unsafe_allow_html=True)
        
        query_vector = model.encode(query).tolist()
        try:
            response = requests.post(f"{ENDEE_URL}/index/{INDEX_NAME}/search", json={
                "vector": query_vector,
                "k": 3
            })
            
            if response.status_code == 200:
                matches = msgpack.unpackb(response.content)
                
                if matches:
                    # Retrieve actual text for each match
                    context_ids = [m[1] for m in matches]
                    context_texts = [st.session_state.chunks_store.get(cid, "[Text Content Missing]") for cid in context_ids]
                    
                    st.subheader("💡 AI Generated Response")
                    answer = generate_answer_mock(query, context_texts)
                    st.info(answer)
                    
                    with st.expander("📚 View All Matched Context Blocks"):
                        for i, (match, txt) in enumerate(zip(matches, context_texts)):
                            st.markdown(f"""
                            <div class="result-card">
                                <b>Source:</b> {match[1]} | <b>Score:</b> {match[0]:.4f}<br/>
                                <p style='color: #ccc; margin-top: 10px;'>{txt}</p>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.warning("Agent found no relevant data in Endee.")
            else:
                st.error("Endee Search Error.")
        except Exception as e:
            st.error(f"Connection failed: {e}")
    else:
        st.markdown(f'<div class="agent-thought">Agent Action: General response.</div>', unsafe_allow_html=True)
        st.write("Hello! I'm your research assistant. Ask me anything technical about your documents.")

st.markdown("---")
st.caption("Powered by Endee Vector Database & Local Session Store")
