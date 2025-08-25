# RAG Project - Version 2

## Components
- `RAG_Search_new2.py`: Enhanced RAG implementation
- `RAG_Search_web2.py`: Improved web interface
- `enhanced_rag_chatbot_faiss.py`: FAISS-based chatbot

## Enhancements
- Improved document processing
- Enhanced caching system
- Better error handling
- Conversation history
- Streamlit UI integration

## Usage
```python
from enhanced_rag_chatbot_faiss import ChatBot
bot = ChatBot()
response = bot.get_response("your question")
```