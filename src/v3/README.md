# RAG Project - Version 3

## Components
- `enhanced_rag_chatbot_faiss_upgraded_v1.py` - Initial v3 implementation
- `enhanced_rag_chatbot_faiss_upgraded_v2.py` - Intermediate improvements
- `enhanced_rag_chatbot_faiss_upgraded_v3_2.py` - Latest enhanced version

## Advanced Features

### Document Processing
- Multi-format document support (PDF, Text, Code)
- OCR capabilities with PyMuPDF and Tesseract
- Intelligent PDF parsing with fallback mechanisms
- Code syntax highlighting and language detection

### Vector Search & Embeddings
- FAISS vector store integration
- Multiple embedding options:
  - Ollama embeddings
  - HuggingFace embeddings
  - Configurable model selection

### Performance
- Asynchronous document processing
- Advanced caching system with:
  - LRU cache
  - Disk cache
  - Document fingerprinting
- Multi-threaded operations

### Intelligence
- Multi-language support
- Context-aware responses
- Source attribution
- Response optimization
- Conversation state management

### Monitoring & Logging
- Progress tracking
- Detailed logging
- Cache statistics
- Performance metrics

## Installation

```bash
pip install -r requirements.txt

# Optional: For GPU support
pip install torch==2.0.1+cu118
```

## Usage

### Basic Usage
```python
from enhanced_rag_chatbot_faiss_upgraded_v3_2 import EnhancedRAGChatbot

# Initialize chatbot
bot = EnhancedRAGChatbot(cache_dir="./cache")

# Configure models
bot.initialize_models(
    embedding_model="all-MiniLM-V6-v2",
    llm_model="llama3.2",
    use_ollama_embeddings=True
)

# Process documents
documents = await bot.process_pdf_folder_async("path/to/docs")

# Query
response = bot.query("Your question here")
```

### Advanced Usage
```python
# With custom progress tracking
def progress_callback(message, progress):
    print(f"{progress}%: {message}")

bot.set_progress_callback(progress_callback)

# Cache management
stats = bot.get_cache_stats()
bot.clear_cache()

# Async document processing
async with bot.session():
    docs = await bot.process_documents_async(documents)
```

## Configuration

### Environment Variables
```env
OLLAMA_BASE_URL=http://localhost:11434
CACHE_DIR=./cache
MAX_WORKERS=4
```

## System Requirements
- Python 3.8+
- 8GB RAM minimum
- Optional: NVIDIA GPU for acceleration
- Storage: 1GB+ for cache and vector stores

## Contributing
See CONTRIBUTING.md for development guidelines.

## License
MIT License