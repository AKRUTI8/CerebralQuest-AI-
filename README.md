# CerebralQuest-AI-
CerebralQuest AI is a research assistant web app built with Streamlit that helps users explore, summarize, and interact with scientific research. It leverages arXiv, RSS feeds, and custom URLs to fetch papers and content, and uses AI models for summarization, keyword extraction, and question answering.
Key Features:

PDF Summarizer: Generate concise bullet-point summaries and extract keywords from uploaded PDFs.

Research Chatbot: Fetch latest research from arXiv, RSS feeds, or custom URLs, and ask questions to get AI-generated answers based on retrieved documents.

Embeddings & Retrieval: Uses SentenceTransformers and FAISS for semantic search over documents.

Summarization & Q/A: Powered by Flan-T5 small for lightweight text generation and summarization.

Downloadable Results: Summaries and answers can be exported as .txt files.

Tech Stack:

Python, Streamlit

PyMuPDF, BeautifulSoup, feedparser

SentenceTransformers, HuggingFace Transformers

FAISS for semantic search

Use Case: Perfect for researchers, students, and AI enthusiasts who want quick insights from scientific papers without reading entire PDFs.
