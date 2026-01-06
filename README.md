## YouTube RAG Chatbot (LangChain)

This project demonstrates a Retrieval-Augmented Generation (RAG) pipeline
that enables querying and summarization of YouTube videos using LLMs.
Notebook used for rapid iteration; pipeline designed to be **framework-agnostic** 
and productionized behind an API.

### Current State
- Implemented as a research prototype in Jupyter Notebook
- Focused on validating chunking, embeddings, retrieval, and prompt design
- API keys are loaded via environment variables

### Tech Stack
- Python, LangChain, FAISS
- OpenAI Embeddings & LLMs
- YouTube API

### Next Steps
- Expose pipeline via FastAPI
- Add authentication & rate limiting
- Build a simple UI

### From the developer
Main focus is learning how to build it, since adding UI around it takes more time 
that I can rather spend on learning LangGraph. Jupyter notebook is just a prototype to show the 
basic workflow of different components combined using Chains.
