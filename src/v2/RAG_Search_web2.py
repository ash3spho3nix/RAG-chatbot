from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import List, Optional
import os
from functools import lru_cache
from langchain_ollama import OllamaEmbeddings, ChatOllama
from RAG_Search_new2 import (
    create_vector_store_from_pdfs,
    load_vector_store,
    RAG_search
)
import markdown2
from bs4 import BeautifulSoup
import logging
from datetime import datetime

os.environ["LANGCHAIN_TRACING_V2"] = "true"
#os.environ["LANGCHAIN_API_KEY"] = ""

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Search System")
app.mount("/templates", StaticFiles(directory="templates"), name="templates")

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
    api_key: Optional[str] = "..."

class VectorStoreRequest(BaseModel):
    pdf_folder: str
    vector_store_path: str

# Conversation history
conversation_history: List[str] = []

# Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    logger.info("Serving index page")
    return FileResponse("templates/index.html")

# Cache configuration
CACHE_SIZE = 5

@lru_cache(maxsize=CACHE_SIZE)
def get_cached_vector_store(vector_path: str):
    logger.info(f"Loading vector store from cache: {vector_path}")
    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        return load_vector_store(vector_path, embeddings)
    except Exception as e:
        logger.error(f"Error loading vector store: {str(e)}")
        raise

@app.post("/vector-store")
async def create_vector_store(request: VectorStoreRequest):
    logger.info(f"Creating vector store from folder: {request.pdf_folder}")
    try:
        await create_vector_store_from_pdfs(
            request.pdf_folder,
            request.vector_store_path
        )
        logger.info("Vector store created successfully")
        return {"message": "Vector store created successfully"}
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add new request model
class LoadVectorStoreRequest(BaseModel):
    vector_store_path: str

@app.post("/load-vector-store")
async def load_vector_store_endpoint(request: LoadVectorStoreRequest):
    logger.info(f"Loading vector store from path: {request.vector_store_path}")
    try:
        # Clear existing cache
        get_cached_vector_store.cache_clear()
        # Load vector store
        logger.info("Loading vector store from path", request.vector_store_path)
        vector_store = get_cached_vector_store(request.vector_store_path)
        logger.info("Vector store loaded successfully")
        return {"message": "Vector store loaded successfully"}
    except Exception as e:
        logger.error(f"Error loading vector store: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    

def format_response(response: str) -> str:
    """Format response with markdown and clean HTML"""
    logger.info("Formatting response")
    """
    Format response with markdown and apply consistent HTML styling
    Handles both markdown and direct HTML content
    """
    logger.info("Formatting response")
    try:
        # Convert markdown to HTML if response contains markdown syntax
        if any(marker in response for marker in ['###', '##', '#', '```', '**', '*', '>']):
            html = markdown2.markdown(response, extras=['tables', 'fenced-code-blocks', 'break-on-newline'])
        else:
            html = response

        # Parse and style HTML
        soup = BeautifulSoup(html, 'html.parser')

        # Define styling rules
        style_rules = {
            'ul, ol': {'class': 'list-disc pl-5 space-y-2 my-3'},
            'h1, h2, h3, h4': {'class': 'font-bold my-3 text-gray-900'},
            'code': {'class': 'bg-gray-100 px-2 py-1 rounded font-mono text-sm'},
            'pre': {'class': 'bg-gray-100 p-4 rounded overflow-x-auto my-4'},
            'p': {'class': 'my-2 text-gray-700 leading-relaxed'},
            'table': {'class': 'min-w-full divide-y divide-gray-200 my-4'},
            'th': {'class': 'px-6 py-3 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider'},
            'td': {'class': 'px-6 py-4 whitespace-nowrap text-sm text-gray-900'},
            'tr': {'class': 'bg-white border-b hover:bg-gray-50'},
            'a': {'class': 'text-blue-600 hover:text-blue-800', 'target': '_blank'},
            'img': {'class': 'max-w-full h-auto rounded-lg my-4'},
            'blockquote': {'class': 'border-l-4 border-gray-300 pl-4 italic text-gray-600 my-3'},
            'strong': {'class': 'font-semibold text-gray-800'},
            'hr': {'class': 'my-4 border-t border-gray-200'},
            'li': {'class': 'mb-2'},
            'div': {'class': 'my-2'}
        }

        # Apply styling rules
        for selector, attributes in style_rules.items():
            for element in soup.select(selector):
                for attr, value in attributes.items():
                    if attr == 'style':
                        current_style = element.get('style', '')
                        element['style'] = f"{current_style}; {value}"
                    else:
                        current_class = element.get('class', [])
                        if isinstance(current_class, list):
                            current_class.extend(value.split())
                        else:
                            current_class = value
                        element['class'] = current_class

        # Add responsive container for tables
        for table in soup.find_all('table'):
            wrapper = soup.new_tag('div')
            wrapper['class'] = 'overflow-x-auto shadow-md rounded-lg my-4'
            table.wrap(wrapper)

        # Format code blocks with syntax highlighting
        for pre in soup.find_all('pre'):
            if 'code' in pre.get('class', []):
                pre['class'] = 'bg-gray-100 p-4 rounded-lg overflow-x-auto my-4'
                for code in pre.find_all('code'):
                    code['class'] = 'language-python'  # Add more languages as needed

        logger.info("Response formatted successfully")
        return str(soup)

    except Exception as e:
        logger.error(f"Error formatting response: {str(e)}")
        logger.debug(f"Original response returned due to error: {response}")
        return response

@app.post("/search")
async def search(request: SearchRequest):
    try:
        result = RAG_search(request.query, conversation_history)
        
        # Format response as markdown
        formatted_result = f"""
### Search Results

{result.get('output', str(result))}

---

**Additional Information:**
- Query Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Source: RAG Search
        """
        
        html_result = format_response(formatted_result)
        conversation_history.append({"user": request.query, "assistant": html_result})
        logger.info(f"Search completed for query: {request.query}")

        return {"response": html_result}
    except Exception as e:
        logger.error(f"Error during search: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info("Starting RAG Search API")
    uvicorn.run(app, host="0.0.0.0", port=8000)