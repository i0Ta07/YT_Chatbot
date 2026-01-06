## YouTube RAG Chatbot (LangChain)

This project demonstrates a Retrieval-Augmented Generation (RAG) pipeline
that enables querying and summarization of YouTube videos using LLMs.
Notebook used for rapid iteration; pipeline designed to be **framework-agnostic** and productionized behind an API.

### Current State
- Implemented as a research prototype in Jupyter Notebook
- Focused on validating chunking, embeddings, retrieval, and prompt design

### Tech Stack
- Python, LangChain, FAISS
- OpenAI Embeddings & LLMs
- YouTube API

### Next Steps
- Expose pipeline via FastAPI
- Add authentication & rate limiting
- Build a simple UI
