# Endee Agentic RAG Assistant ��🧠

This project is a high-performance **Agentic RAG (Retrieval Augmented Generation)** system built using the **Endee Vector Database**. It goes beyond simple vector search by implementing a full document-to-answer pipeline with autonomous agent logic.

## 🌟 Advanced Features
- **Autonomous Agent Routing**: The system acts as an agent, deciding whether to query the vector database (Endee) or answer directly based on the query's complexity.
- **Smart PDF/Text Ingestion**: Built-in processor to handle PDF uploads, automated text extraction, and recursive chunking.
- **SIMD Optimized Retrieval**: Utilizing Endee's C++ core with NEON (Apple Silicon) acceleration for sub-millisecond similarity search.
- **Explainable "Thoughts"**: The UI displays the agent's internal thought process (Reasoning) before providing answers.

## 🏗 System Architecture
1. **Data Layer**: Endee Vector Database (C++) storing 384-dimensional embeddings.
2. **Logic Layer**: Python backend handling `SentenceTransformers` for embedding and `RecursiveCharacterTextSplitter` for chunking.
3. **Agent Layer**: Routing logic that determines context retrieval needs.
4. **UI Layer**: Streamlit dashboard with a premium dark-mode interface.

## � Getting Started

### 1. Build & Start Endee
```bash
# Build the server (Apple Silicon optimized)
cd server && ./install.sh --release --neon

# Start the server (runs on port 8080)
./run_project.sh
```

### 2. Using the Dashboard
1.  Open **[http://localhost:8501](http://localhost:8501)**.
2.  Click **"Initialize Master Index"** in the sidebar.
3.  **Upload a PDF** file from your local machine.
4.  Click **"Knowledge Ingest"** to chunk and store the data in Endee.
5.  Ask questions! The agent will search Endee and synthesize an answer.

## � Technical Approach
- **Embeddings**: `all-MiniLM-L6-v2` - chosen for its optimal balance of speed and semantic accuracy.
- **Vector Metric**: `Cosine Similarity` - used for robust semantic comparison.
- **Chunking Strategy**: `RecursiveCharacterTextSplitter` with 500-character chunks and 50-character overlap to preserve semantic context at boundaries.
- **Performance**: Leverages Endee's SIMD (NEON) implementation to ensure that even with large document sets, retrieval remains near-instant.

---
*Developed for the Endee Labs AI/ML Project Evaluation.*
