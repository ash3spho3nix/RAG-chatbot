from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from functools import lru_cache
import uvicorn
import os
from typing import List, Optional
from datetime import datetime

# Import RAG components
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS

from RAG_Search_new import *
from RAG_Search_new import RAG_search, create_retriever_tool_from_vector
# ...existing imports from RAG_Search_new.py...

app = FastAPI(title="RAG Search API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class SearchRequest(BaseModel):
    query: str
    api_key: Optional[str] = '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'

class VectorStoreConfig(BaseModel):
    folder_path: str
    index_name: str = 'cpm-index'
    model_name: str = 'nomic-embed-text'

# Cache configuration
CACHE_SIZE = 5

@lru_cache(maxsize=CACHE_SIZE)
def get_vector_store(folder_path: str, index_name: str):
    """Cached vector store loader"""
    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        return FAISS.load_local(
            folder_path=folder_path,
            embeddings=embeddings,
            index_name=index_name,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading vector store: {str(e)}")

# Store conversation history in memory (consider using Redis for production)
conversation_history: List[dict] = []

@app.get("/", response_class=HTMLResponse)
async def get_html():
    """Serve the HTML frontend"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAG Search Interface</title>
        <style>
            body { max-width: 800px; margin: 0 auto; padding: 20px; font-family: Arial; }
            .container { display: flex; flex-direction: column; gap: 20px; }
            textarea { width: 100%; height: 150px; }
            .history { max-height: 300px; overflow-y: auto; }
            .message { margin: 10px 0; padding: 10px; border-radius: 5px; }
            .user { background: #e3f2fd; }
            .assistant { background: #f5f5f5; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>RAG Search Interface</h1>
            
            <div>
                <h3>Vector Store Configuration</h3>
                <input type="text" id="vectorPath" placeholder="Vector store path">
                <button onclick="loadVectorStore()">Load Vector Store</button>
            </div>

            <div>
                <h3>Search Query</h3>
                <textarea id="query" placeholder="Enter your query here..."></textarea>
                <button onclick="search()">Search</button>
            </div>

            <div class="history" id="history">
                <h3>Conversation History</h3>
            </div>
        </div>

        <script>
        async function search() {
            const query = document.getElementById('query').value;
            try {
                const response = await fetch('/search', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query })
                });
                const result = await response.json();
                updateHistory(query, result.response);
                document.getElementById('query').value = '';
            } catch (error) {
                alert('Error: ' + error);
            }
        }

        function updateHistory(query, response) {
            const history = document.getElementById('history');
            history.innerHTML += `
                <div class="message user">User: ${query}</div>
                <div class="message assistant">Assistant: ${response}</div>
            `;
            history.scrollTop = history.scrollHeight;
        }

        async function loadVectorStore() {
            const path = document.getElementById('vectorPath').value;
            try {
                const response = await fetch('/load-vector-store', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ folder_path: path })
                });
                const result = await response.json();
                alert(result.message);
            } catch (error) {
                alert('Error: ' + error);
            }
        }
        </script>
    </body>
    </html>
    """

@app.post("/load-vector-store")
async def load_vector_store_endpoint(config: VectorStoreConfig):
    """Endpoint to load vector store"""
    try:
        get_vector_store.cache_clear()  # Clear existing cache
        get_vector_store(config.folder_path, config.index_name)
        return {"message": "Vector store loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search_endpoint(request: SearchRequest, background_tasks: BackgroundTasks):
    """Endpoint to perform RAG search"""
    try:
        if request.api_key:
            os.environ["LANGCHAIN_API_KEY"] = request.api_key

        vector_store = get_vector_store('vector_path', 'cpm-index')
        local_search = create_retriever_tool_from_vector(vector_store)
        
        model = ChatOllama(model="llama3.1", temperature=0.7)
        output = RAG_search(request.query, [msg["content"] for msg in conversation_history])

        # Update conversation history
        conversation_history.append({
            "role": "user",
            "content": request.query,
            "timestamp": datetime.now().isoformat()
        })

        # Limit history size
        if len(conversation_history) > 100:
            conversation_history.pop(0)

        return {"response": str(output)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversation-history")
async def get_conversation_history():
    """Get conversation history"""
    return conversation_history

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)