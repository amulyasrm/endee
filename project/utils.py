from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import io

def extract_text_from_pdf(file_bytes):
    pdf = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in pdf.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_text(text)

def generate_answer_mock(query, context_texts):
    """
    Takes the actual text from documents and 'answers' the query.
    """
    if not context_texts:
        return "I couldn't find any specific information in the documents."
    
    # In this mock, we return the best matching text chunk directly
    # This simulates a RAG system returning the exact factual context.
    best_match = context_texts[0]
    
    # Project specific info catch
    if "project" in query.lower() or "explain" in query.lower():
        return "This project is an Advanced Agentic RAG system. It uses Endee as a vector database for high-performance semantic search and LLM-based reasoning."

    return f"I found the following information in the documents:\n\n\"{best_match.strip()}\"\n\n---\n*Does this answer your question about '{query}'?*"
