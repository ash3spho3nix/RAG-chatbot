# RAG Project - Version 1

## Components
- `RAG_Search_new.py`: Core RAG implementation
- `RAG_Search_web.py`: FastAPI web interface
- `response_functions.py`: Helper utilities

## Features
- Basic RAG implementation
- PDF document processing
- FAISS vector store integration
- Simple web API interface

## Usage
```python
from RAG_Search_new import RAG_search
response = RAG_search("your question")
```