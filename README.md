# RAG (Retrieval Augmented Generation) Project

## Project Overview
My first experiment with RAG based chatbot, with an attempt to include features to make it faster, accurate and more robust.
Advanced document retrieval and question-answering system using RAG architecture, combining FAISS vector storage with Ollama LLM capabilities.

## Versions
- **Version 1**: Basic RAG implementation with FastAPI (Desktop version also available)
- **Version 2**: Enhanced chatbot with FAISS (Desktop version also available)
- **Version 3**: Advanced features and optimizations(v0, v1 , v2) [Streamlit versions]

## Version 3
### Advanced Features

#### Document Processing
- Multi-format document support (PDF, Text, Code)
- OCR capabilities with PyMuPDF and Tesseract
- Intelligent PDF parsing with fallback mechanisms
- Code syntax highlighting and language detection

#### Vector Search & Embeddings
- FAISS vector store integration
- Multiple embedding options:
  - Ollama embeddings
  - HuggingFace embeddings
  - Configurable model selection

#### Performance
- Asynchronous document processing
- Advanced caching system with:
  - LRU cache
  - Disk cache
  - Document fingerprinting
- Multi-threaded operations

#### Intelligence
- Multi-language support
- Context-aware responses
- Source attribution
- Response optimization
- Conversation state management

#### Monitoring & Logging
- Progress tracking
- Detailed logging
- Cache statistics
- Performance metrics


## 🌟 Key Features of the Project
- **Multi-Version Support**: Three distinct versions with incremental improvements
- **Document Processing**: Handle PDFs, TXTs, and other text-based formats
- **Vector Storage**: FAISS-based efficient similarity search
- **Multiple Interfaces**: FastAPI and Streamlit implementations
- **Async Processing**: Enhanced performance with asynchronous operations
- **Caching System**: Optimized response times
- **Testing Suite**: Comprehensive unit and integration tests

## 📁 Project Structure
```
RAG_new/
├── src/
│   ├── v1/         # Base implementation
│   ├── v2/         # Enhanced features
│   └── v3/         # Latest upgrades
├── data/           # Document and vector stores
├── tests/          # Testing suite
├── config/         # Configuration files
├── utils/          # Helper utilities
└── web/           # Web interface assets
```

## 🚀 Quick Start
```bash
# Clone repository
git clone https://github.com/yourusername/RAG_new.git

# Install dependencies
pip install -r requirements.txt

# Run tests
python run_tests.py

# Start web interface
python src/v1/RAG_Search_web.py
```

## 🔧 Configuration
1. Set up Ollama LLM
2. Configure vector store path in `config/settings.py`
3. Add documents to `data/documents/`

## 💻 Usage Examples
```python
from src.v1.RAG_Search_new import RAG_search

# Simple query
response = RAG_search("How does RAG work?")

# Document ingestion
from src.v1.RAG_Search_new import create_vector_store_from_pdfs
create_vector_store_from_pdfs("path/to/docs")
```

## 🧪 Testing
```bash
# Run all tests
pytest tests -v

# Run specific test category
pytest tests/unit -v
pytest tests/integration -v
```

## 📚 Version Details
- **V1**: Basic RAG implementation with FastAPI
- **V2**: Enhanced chatbot with FAISS
- **V3**: Advanced features and optimizations

## 📋 Requirements
- check requirements file

## 📜 License
MIT License

## 👥 Contributing
See CONTRIBUTING.md for guidelines.