import streamlit as st
import os
import tempfile
from pathlib import Path
import shutil
from typing import List, Dict, Any, Optional, Tuple
import pickle
import asyncio
import logging
import hashlib
import json
from datetime import datetime
import concurrent.futures
from functools import lru_cache
import time
import validators
# Updated LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.schema import Document

# Using FAISS instead of ChromaDB to avoid ONNX issues
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader
)
from langchain_ollama import OllamaEmbeddings, OllamaLLM

# Updated HuggingFace embeddings import
from langchain_huggingface import HuggingFaceEmbeddings

# Alternative PDF loaders
try:
    from langchain_community.document_loaders import PyPDFLoader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

try:
    from langchain_community.document_loaders import PyPDF2Loader
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

import PyPDF2
import chardet
import aiofiles

from response_functions import (
    ResponseOptimizer,
    VectorStoreManager,
    StreamingLLMResponse
)

# Add to existing imports
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt
from pdf2image import convert_from_path
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import validators

class EnhancedRAGChatbot:
    def __init__(self, cache_dir: str = "./cache", max_workers: int = 4):
        """Initialize the Enhanced RAG Chatbot with caching and async support"""
        # Setup logging
        self.setup_logging()
        
        # Initialize cache
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Core components
        self.embeddings = None
        self.vectorstore = None
        self.qa_chain = None
        self.documents = []
        self.llm = None

        self.vs_manager = VectorStoreManager("./vectorstores")
        # Add response optimizer
        self.response_optimizer = ResponseOptimizer(
            embedding_model="all-MiniLM-L6-v2",  # Fast, lightweight model
            vectorstore_dir="./vectorstores",
            cache_size=1000
        )

        # Async processing
        self.max_workers = max_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
        # Cache dictionaries
        self.document_cache = {}
        self.embedding_cache = {}
        
        self.logger.info("Enhanced RAG Chatbot initialized")
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Setup logging configuration
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'rag_chatbot.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Also setup Streamlit progress logging
        self.progress_callback = None
    
    def set_progress_callback(self, callback):
        """Set callback for Streamlit progress updates"""
        self.progress_callback = callback
    
    def update_progress(self, message: str, progress: float = None):
        """Update progress in both logger and Streamlit"""
        self.logger.info(message)
        if self.progress_callback:
            self.progress_callback(message, progress)
    
    def get_file_hash(self, file_path: Path) -> str:
        """Generate hash for file caching"""
        try:
            with open(file_path, 'rb') as f:
                file_content = f.read()
                return hashlib.md5(file_content).hexdigest()
        except Exception as e:
            self.logger.error(f"Error generating hash for {file_path}: {e}")
            return str(file_path)
    
    def get_folder_hash(self, folder_path: Path) -> str:
        """Generate hash for folder based on all PDF files"""
        pdf_files = sorted(folder_path.glob("*.pdf"))
        if not pdf_files:
            return ""
        
        hash_content = ""
        for pdf_file in pdf_files:
            try:
                stat = pdf_file.stat()
                hash_content += f"{pdf_file.name}_{stat.st_size}_{stat.st_mtime}_"
            except Exception:
                hash_content += f"{pdf_file.name}_"
        
        return hashlib.md5(hash_content.encode()).hexdigest()
    
    def load_cache(self, cache_key: str) -> Optional[Dict]:
        """Load cached data"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                    self.logger.info(f"Cache hit: {cache_key}")
                    return data
            except Exception as e:
                self.logger.warning(f"Failed to load cache {cache_key}: {e}")
                # Remove corrupted cache file
                try:
                    cache_file.unlink()
                except:
                    pass
        return None
    
    def save_cache(self, cache_key: str, data: Dict):
        """Save data to cache"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            self.logger.info(f"Cache saved: {cache_key}")
        except Exception as e:
            self.logger.error(f"Failed to save cache {cache_key}: {e}")
    
    def initialize_models(self, embedding_model: str = "all-Minilm-V6-v2", 
                         llm_model: str = "llama3.2", use_ollama_embeddings: bool = False):

        """Initialize models with fallback options"""
        try:
            self.update_progress("Initializing models...")
            
            # Initialize LLM
            self.update_progress(f"Loading LLM model: {llm_model}")

            # # Replace HuggingFace embeddings with OptimizedEmbeddings
            # if not use_ollama_embeddings:
            #     self.embeddings = HuggingFaceEmbeddings(
            #         model_name="sentence-transformers/all-MiniLM-L6-v2",
            #         model_kwargs={'device': 'cpu'},
            #         encode_kwargs={'normalize_embeddings': True}
            #     )
            # else:
            #     self.embeddings = OptimizedEmbeddings(
            #         model_name=embedding_model,
            #         cache_size=1000
            #     )
                
            self.llm = OllamaLLM(
                model=llm_model,
                temperature=0.1,
                base_url="http://localhost:11434"
            )
            # self.streaming_llm = StreamingLLMResponse(
            #     self.llm,
            #     max_context_length= 4000,
            #     max_tokens= 1000,
            # )
            self.update_progress("LLM model initialized successfully!")
            
            # Initialize embeddings with fallback
            if use_ollama_embeddings:
                self.update_progress(f"Loading Ollama embedding model: {embedding_model}")
                try:
                    self.embeddings = OllamaEmbeddings(
                        model=embedding_model,
                        base_url="http://localhost:11434"
                    )
                    # Test the embedding model
                    test_embedding = self.embeddings.embed_query("test")
                    self.update_progress("Ollama embeddings initialized successfully!")
                except Exception as e:
                    if "onnxruntime" in str(e).lower():
                        self.logger.error("ONNX error with Ollama embeddings. Switching to HuggingFace...")
                        use_ollama_embeddings = False
                    else:
                        raise e
            
            if not use_ollama_embeddings:
                self.update_progress("Loading HuggingFace embeddings (no ONNX required)")
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                # Test embedding
                test_embedding = self.embeddings.embed_query("test")
                self.update_progress("HuggingFace embeddings initialized successfully!")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing models: {str(e)}")
            return False
    
    async def process_pdf_file_async(self, pdf_path: Path) -> List[Document]:
        """Asynchronously process a single PDF file"""
        self.update_progress(f"Processing: {pdf_path.name}")
        
        # Check cache first
        file_hash = self.get_file_hash(pdf_path)
        cache_key = f"pdf_{file_hash}"
        cached_data = self.load_cache(cache_key)
        
        if cached_data:
            self.update_progress(f"Using cached data for: {pdf_path.name}")
            documents = []
            for doc_data in cached_data['documents']:
                doc = Document(
                    page_content=doc_data['page_content'],
                    metadata=doc_data['metadata']
                )
                documents.append(doc)
            return documents
        
        # Process PDF if not cached
        def process_pdf():
            return self._load_pdf_with_fallback(str(pdf_path), pdf_path.name)
        
        loop = asyncio.get_event_loop()
        documents = await loop.run_in_executor(self.executor, process_pdf)
        
        # Cache the results
        if documents:
            cache_data = {
                'documents': [
                    {
                        'page_content': doc.page_content,
                        'metadata': doc.metadata
                    }
                    for doc in documents
                ],
                'processed_at': datetime.now().isoformat(),
                'file_size': pdf_path.stat().st_size
            }
            self.save_cache(cache_key, cache_data)
        
        self.update_progress(f"Extracted {len(documents)} chunks from: {pdf_path.name}")
        return documents
    
    async def process_pdf_folder_async(self, folder_path: str) -> List[Document]:
        """Asynchronously process all PDFs in a folder"""
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        # Find all PDF files (no size limit)
        pdf_files = list(folder.glob("*.pdf"))
        if not pdf_files:
            self.logger.warning(f"No PDF files found in {folder_path}")
            return []
        
        self.update_progress(f"Found {len(pdf_files)} PDF files to process")
        
        # Check if we have cached folder data
        folder_hash = self.get_folder_hash(folder)
        folder_cache_key = f"folder_{folder_hash}"
        cached_folder_data = self.load_cache(folder_cache_key)
        
        if cached_folder_data:
            self.update_progress("Using cached folder data")
            documents = []
            for doc_data in cached_folder_data['documents']:
                doc = Document(
                    page_content=doc_data['page_content'],
                    metadata=doc_data['metadata']
                )
                documents.append(doc)
            return documents
        
        # Process PDFs concurrently
        self.update_progress("Processing PDFs concurrently...")
        tasks = [self.process_pdf_file_async(pdf_file) for pdf_file in pdf_files]
        
        all_documents = []
        completed = 0
        
        for task in asyncio.as_completed(tasks):
            documents = await task
            all_documents.extend(documents)
            completed += 1
            progress = completed / len(pdf_files)
            self.update_progress(f"Progress: {completed}/{len(pdf_files)} PDFs processed", progress)
        
        # Cache folder results
        if all_documents:
            folder_cache_data = {
                'documents': [
                    {
                        'page_content': doc.page_content,
                        'metadata': doc.metadata
                    }
                    for doc in all_documents
                ],
                'processed_at': datetime.now().isoformat(),
                'total_files': len(pdf_files),
                'total_documents': len(all_documents)
            }
            self.save_cache(folder_cache_key, folder_cache_data)
        
        self.update_progress(f"Successfully processed {len(all_documents)} documents from {len(pdf_files)} PDFs")
        return all_documents
    
    def load_pdf_documents(self, pdf_files: List) -> List[Document]:
        """Load and process uploaded PDF documents"""
        documents = []
        total_files = len(pdf_files)
        
        for i, pdf_file in enumerate(pdf_files):
            try:
                self.update_progress(f"Processing uploaded file: {pdf_file.name} ({i+1}/{total_files})")
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    # Reset file pointer
                    pdf_file.seek(0)
                    tmp_file.write(pdf_file.read())
                    tmp_file_path = tmp_file.name
                
                # Check cache for uploaded file (using content hash)
                pdf_file.seek(0)
                content = pdf_file.read()
                content_hash = hashlib.md5(content).hexdigest()
                cache_key = f"upload_{content_hash}"
                
                cached_data = self.load_cache(cache_key)
                if cached_data:
                    self.update_progress(f"Using cached data for uploaded file: {pdf_file.name}")
                    for doc_data in cached_data['documents']:
                        doc = Document(
                            page_content=doc_data['page_content'],
                            metadata=doc_data['metadata']
                        )
                        documents.append(doc)
                else:
                    pdf_docs = self._load_pdf_with_fallback(tmp_file_path, pdf_file.name)
                    documents.extend(pdf_docs)
                    
                    # Cache the results
                    if pdf_docs:
                        cache_data = {
                            'documents': [
                                {
                                    'page_content': doc.page_content,
                                    'metadata': doc.metadata
                                }
                                for doc in pdf_docs
                            ],
                            'processed_at': datetime.now().isoformat()
                        }
                        self.save_cache(cache_key, cache_data)
                
                os.unlink(tmp_file_path)
                progress = (i + 1) / total_files
                self.update_progress(f"Completed: {pdf_file.name}", progress)
                
            except Exception as e:
                self.logger.error(f"Error loading PDF {pdf_file.name}: {str(e)}")
        
        return documents
    
    def _load_pdf_with_fallback(self, file_path: str, file_name: str) -> List[Document]:
        """Enhanced PDF loading with multiple fallbacks and equation/figure extraction"""
        documents = []
        
        # Try PyMuPDF (fitz) first for advanced features
        try:
            pdf_document = fitz.open(file_path)
            
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                
                # Extract text content
                text = page.get_text()
                
                # Extract images
                images = page.get_images()
                image_list = []
                
                for img_index, img in enumerate(images):
                    try:
                        # Get image data
                        xref = img[0]
                        base_image = pdf_document.extract_image(xref)
                        image_data = base_image["image"]
                        
                        # Convert to PIL Image for OCR
                        image = Image.open(io.BytesIO(image_data))
                        
                        # Perform OCR on image
                        image_text = pytesseract.image_to_string(image)
                        
                        # Save image data
                        image_b64 = base64.b64encode(image_data).decode()
                        image_list.append({
                            'data': image_b64,
                            'ocr_text': image_text,
                            'index': img_index
                        })
                    except Exception as e:
                        self.logger.warning(f"Error processing image {img_index} on page {page_num}: {e}")
            
                # Extract equations (MathML format)
                equations = []
                for block in page.get_text("dict")["blocks"]:
                    if block.get("type") == 1:  # Type 1 blocks are usually equations
                        equations.append(block.get("text", ""))
                
                # Create document with enhanced metadata
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": file_name,
                        "type": "pdf",
                        "page": page_num + 1,
                        "loader": "PyMuPDF",
                        "images": image_list,
                        "equations": equations,
                        "has_figures": len(image_list) > 0,
                        "has_equations": len(equations) > 0
                    }
                )
                documents.append(doc)
                
            if documents:
                return documents
            
        except Exception as e:
            self.logger.warning(f"PyMuPDF extraction failed for {file_name}: {str(e)}")
    
        # Fallback 1: Manual PyPDF2 extraction
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        text = page.extract_text()
                        if text.strip():
                            doc = Document(
                                page_content=text,
                                metadata={
                                    "source": file_name,
                                    "type": "pdf",
                                    "page": page_num + 1,
                                    "loader": "PyPDF2_manual"
                                }
                            )
                            documents.append(doc)
                    except Exception as e:
                        self.logger.warning(f"Error extracting page {page_num + 1} from {file_name}: {e}")
                        continue
                
                if documents:
                    return documents
                    
        except Exception as e:
            self.logger.warning(f"PyPDF2 manual extraction failed: {str(e)}")
    
        # Fallback 2: PyPDF2Loader
        if PYPDF2_AVAILABLE:
            try:
                loader = PyPDF2Loader(file_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = file_name
                    doc.metadata["type"] = "pdf"
                    doc.metadata["loader"] = "PyPDF2"
                return docs
            except Exception as e:
                self.logger.warning(f"PyPDF2Loader failed for {file_name}: {str(e)}")
        
        # Method 3: Try PyPDFLoader
        if PYPDF_AVAILABLE:
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = file_name
                    doc.metadata["type"] = "pdf"
                    doc.metadata["loader"] = "PyPDF"
                return docs
            except Exception as e:
                self.logger.warning(f"PyPDFLoader failed for {file_name}: {str(e)}")
        
        self.logger.error(f"All PDF loading methods failed for {file_name}")
        return []
    
    def load_code_documents(self, code_files: List) -> List[Document]:
        """Load and process code files with caching"""
        documents = []
        total_files = len(code_files)
        
        code_extensions = {
            '.py', '.js', '.java', '.cpp', '.c', '.h', '.hpp', '.cs', '.rb', 
            '.go', '.rs', '.php', '.swift', '.kt', '.ts', '.jsx', '.tsx', 
            '.vue', '.html', '.css', '.scss', '.less', '.sql', '.sh', '.bash',
            '.yaml', '.yml', '.json', '.xml', '.md', '.txt', '.r', '.scala',
            '.dart', '.lua', '.perl', '.pl'
        };
        
        for i, code_file in enumerate(code_files):
            try:
                self.update_progress(f"Processing code file: {code_file.name} ({i+1}/{total_files})")
                
                file_extension = Path(code_file.name).suffix.lower()
                
                if file_extension in code_extensions or file_extension == '':
                    # Check cache
                    code_file.seek(0)
                    content = code_file.read()
                    content_hash = hashlib.md5(content).hexdigest()
                    cache_key = f"code_{content_hash}"
                    
                    cached_data = self.load_cache(cache_key)
                    if cached_data:
                        self.update_progress(f"Using cached data for: {code_file.name}")
                        doc = Document(
                            page_content=cached_data['content'],
                            metadata=cached_data['metadata']
                        )
                        documents.append(doc)
                        continue
                    
                    # Process file
                    if isinstance(content, bytes):
                        encoding = chardet.detect(content)['encoding'] or 'utf-8'
                        try:
                            content = content.decode(encoding)
                        except UnicodeDecodeError:
                            content = content.decode('utf-8', errors='ignore')
                    
                    metadata = {
                        "source": code_file.name,
                        "type": "code",
                        "extension": file_extension,
                        "language": self.detect_language(file_extension)
                    }
                    
                    doc = Document(page_content=content, metadata=metadata)
                    documents.append(doc)
                    
                    # Cache the result
                    cache_data = {
                        'content': content,
                        'metadata': metadata,
                        'processed_at': datetime.now().isoformat()
                    }
                    self.save_cache(cache_key, cache_data)
                else:
                    self.logger.warning(f"Unsupported file type: {code_file.name}")
                
                progress = (i + 1) / total_files
                self.update_progress(f"Completed: {code_file.name}", progress)
                    
            except Exception as e:
                self.logger.error(f"Error loading code file {code_file.name}: {str(e)}")
        
        return documents
    
    def detect_language(self, extension: str) -> str:
        """Detect programming language from file extension"""
        language_map = {
            '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
            '.java': 'java', '.cpp': 'cpp', '.c': 'c', '.h': 'c',
            '.cs': 'csharp', '.rb': 'ruby', '.go': 'go', '.rs': 'rust',
            '.php': 'php', '.swift': 'swift', '.kt': 'kotlin',
            '.jsx': 'jsx', '.tsx': 'tsx', '.vue': 'vue',
            '.html': 'html', '.css': 'css', '.scss': 'scss',
            '.sql': 'sql', '.sh': 'bash', '.bash': 'bash',
            '.yaml': 'yaml', '.yml': 'yaml', '.json': 'json',
            '.xml': 'xml', '.md': 'markdown', '.r': 'r',
            '.scala': 'scala', '.dart': 'dart', '.lua': 'lua'
        }
        return language_map.get(extension, 'text')
    
    async def process_documents_async(self, documents: List[Document]) -> bool:
        """Process documents and create FAISS vector store with async support"""
        if not documents:
            self.logger.error("No documents to process")
            return False
        
        try:
            self.update_progress("Splitting documents into chunks...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            # Split documents in executor to avoid blocking
            loop = asyncio.get_event_loop()
            splits = await loop.run_in_executor(
                self.executor, 
                text_splitter.split_documents, 
                documents
            )
            
            self.update_progress(f"Created {len(splits)} text chunks")
            
            self.update_progress("Creating FAISS vector store...")
            
            # Check if we have cached embeddings for this document set
            doc_hash = hashlib.md5(str([doc.page_content for doc in splits]).encode()).hexdigest()
            vectorstore_cache_key = f"vectorstore_{doc_hash}"
            
            cached_vectorstore = self.load_cache(vectorstore_cache_key)
            if cached_vectorstore and os.path.exists("./faiss_index"):
                self.update_progress("Loading cached vector store...")
                try:
                    self.vectorstore = FAISS.load_local(
                        "./faiss_index", 
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    self.update_progress("Cached vector store loaded successfully!")
                except Exception as e:
                    self.logger.warning(f"Failed to load cached vectorstore: {e}")
                    cached_vectorstore = None
            
            if not cached_vectorstore:
                # Create new vector store
                def create_vectorstore():
                    return FAISS.from_documents(
                        documents=splits,
                        embedding=self.embeddings
                    )
                
                self.vectorstore = await loop.run_in_executor(
                    self.executor,
                    create_vectorstore
                )
                
                # Save FAISS index
                self.vectorstore.save_local("./faiss_index")
                
                # Cache vectorstore metadata
                cache_data = {
                    'created_at': datetime.now().isoformat(),
                    'num_documents': len(splits),
                    'doc_hash': doc_hash
                }
                self.save_cache(vectorstore_cache_key, cache_data)
                
                self.update_progress("FAISS vector store created and cached successfully!")
            
            # Create QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 5}
                ),
                return_source_documents=True
            )
            
            self.documents = documents
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing documents: {str(e)}")
            return False
    
    def process_documents(self, documents: List[Document]) -> bool:
        """Synchronous wrapper for document processing"""
        try:
            # Run async function in new event loop
            return asyncio.run(self.process_documents_async(documents))
        except Exception as e:
            self.logger.error(f"Error in sync document processing: {str(e)}")
            return False
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system with caching"""
        if not self.qa_chain:
            return {"error": "No documents loaded or processed"}
        
        if not self.vectorstore or not self.embeddings:
            return {"error": "Components not initialized. Please initialize models first."}
        
        # Check query cache
        query_hash = hashlib.md5(question.encode()).hexdigest()
        query_cache_key = f"query_{query_hash}"
        cached_result = self.load_cache(query_cache_key)
        
        if cached_result:
            self.logger.info(f"Using cached result for query: {question[:50]}...")
            return cached_result['result']
        
        try:
            start_time = time.time()
            result = self.qa_chain.invoke({"query": question})
            end_time = time.time()
            
            response = {
                "answer": result["result"],
                "source_documents": result.get("source_documents", []),
                "processing_time": end_time - start_time
            }
            
            # Cache the result (but not the source documents to save space)
            cache_data = {
                'result': {
                    "answer": result["result"],
                    "source_documents": result.get("source_documents", [])
                },
                'cached_at': datetime.now().isoformat(),
                'processing_time': end_time - start_time
            }
            self.save_cache(query_cache_key, cache_data)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error querying: {str(e)}")
            return {"error": f"Error querying: {str(e)}"}
    
    def clear_cache(self):
        """Clear all cached data"""
        try:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir()
                self.logger.info("Cache cleared successfully")
                return True
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            cache_files = list(self.cache_dir.glob("*.pkl"))
            total_size = sum(f.stat().st_size for f in cache_files)
            
            stats = {
                'cache_files': len(cache_files),
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'cache_dir': str(self.cache_dir)
            }
            
            # Categorize cache files
            categories = {'pdf': 0, 'upload': 0, 'code': 0, 'folder': 0, 'vectorstore': 0, 'query': 0, 'other': 0}
            for f in cache_files:
                name = f.stem
                categorized = False
                for category in categories:
                    if name.startswith(category):
                        categories[category] += 1
                        categorized = True
                        break
                if not categorized:
                    categories['other'] += 1
            
            stats['categories'] = categories
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting cache stats: {e}")
            return {}

    def save_current_state(self, name: str):
        """Save current vectorstore state with validation"""
        if not self.vectorstore or not self.documents:
            self.logger.error("No vectorstore or documents to save")
            return False
        
        if not self.embeddings:
            self.logger.error("Embeddings not initialized")
            return False
        
        metadata = {
            "last_updated": datetime.now().isoformat(),
            "num_documents": len(self.documents),
            "embedding_model": self.embeddings.model_name if hasattr(self.embeddings, 'model_name') else str(self.embeddings),
            "document_types": {}
        }
        
        # Add document type statistics
        for doc in self.documents:
            doc_type = doc.metadata.get('type', 'unknown')
            metadata["document_types"][doc_type] = metadata["document_types"].get(doc_type, 0) + 1
        
        return self.vs_manager.save_vectorstore(
            self.vectorstore,
            self.documents,
            name,
            metadata
        )

    def load_saved_state(self, name: str):
        """Load saved vectorstore state and reconnect QA chain"""
        result = self.vs_manager.load_vectorstore(name, self.embeddings)
        if result:
            self.vectorstore, self.documents, self.metadata = result
            
            # Reconnect QA chain with loaded vectorstore
            if self.llm:
                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=self.vectorstore.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 5}
                    ),
                    return_source_documents=True
                )
                return True
        return False
    
    def scrape_website(self, url: str, max_depth: int = 2) -> List[Document]:
        """Scrape website content and convert to documents"""
        try:
            self.logger.info(f"Starting web scraping for: {url}")
            
            def is_valid_url(url: str) -> bool:
                return validators.url(url)
            
            def get_base_domain(url: str) -> str:
                return urlparse(url).netloc
            
            def scrape_page(url: str, depth: int = 0, visited: set = None) -> List[Document]:
                if visited is None:
                    visited = set()
                
                if depth > max_depth or url in visited:
                    return []
                    
                visited.add(url)
                documents = []
                
                try:
                    response = requests.get(url, timeout=10)
                    response.raise_for_status()
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()
                    
                    # Extract text content
                    text = soup.get_text(separator='\n')
                    
                    # Create document
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": url,
                            "type": "webpage",
                            "title": soup.title.string if soup.title else url,
                            "depth": depth
                        }
                    )
                    documents.append(doc)
                    
                    # Find and process links if not at max depth
                    if depth < max_depth:
                        base_domain = get_base_domain(url)
                        for link in soup.find_all('a'):
                            href = link.get('href')
                            if href:
                                next_url = urljoin(url, href)
                                if (is_valid_url(next_url) and 
                                    get_base_domain(next_url) == base_domain and 
                                    next_url not in visited):
                                    documents.extend(scrape_page(next_url, depth + 1, visited))
                    
                    return documents
                    
                except Exception as e:
                    self.logger.error(f"Error scraping {url}: {str(e)}")
                    return documents
            
            # Start scraping from root URL
            if not is_valid_url(url):
                raise ValueError("Invalid URL provided")
                
            documents = scrape_page(url)
            self.logger.info(f"Scraped {len(documents)} pages from {url}")
            return documents
            
        except Exception as e:
            self.logger.error(f"Web scraping failed: {str(e)}")
            return []
    
def main():
    
    st.set_page_config(
        page_title="Enhanced RAG Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    # Theme selection and page config
    theme = st.sidebar.radio(
        "Select Theme",
        ["Light Classic", "Dark Modern", "Light Modern"],
        key="theme_selection"
    )


    # Theme-based CSS
    dark_modern = """
    <style>
    .main { background: linear-gradient(120deg, #1a1a1a 0%, #2d2d2d 100%); color: #e0e0e0; }
    .stApp { background: linear-gradient(120deg, #1a1a1a 0%, #2d2d2d 100%); }
    .css-1d391kg, .css-12oz5g7 {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 20px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stButton>button {
        background: linear-gradient(90deg, #00c853 0%, #64dd17 100%);
        color: white;
        border-radius: 25px;
        padding: 10px 25px;
        border: none;
        box-shadow: 0 4px 15px rgba(0,200,83,0.2);
    }
    .chat-message {
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .user-message { background: linear-gradient(120deg, #1565c0 0%, #1976d2 100%); color: white; }
    .bot-message { background: linear-gradient(120deg, #2e7d32 0%, #388e3c 100%); color: white; }
    </style>
    """

    light_modern = """
    <style>
    .main { background: linear-gradient(120deg, #f5f5f5 0%, #ffffff 100%); }
    .stApp { background: linear-gradient(120deg, #f5f5f5 0%, #ffffff 100%); }
    .css-1d391kg, .css-12oz5g7 {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background: linear-gradient(90deg, #2196f3 0%, #64b5f6 100%);
        color: white;
        border-radius: 25px;
        padding: 10px 25px;
        border: none;
    }
    .chat-message {
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .user-message { background: #e3f2fd; color: #1565c0; }
    .bot-message { background: #f1f8e9; color: #2e7d32; }
    </style>
    """

    light_classic = """
    <style>
    .stApp { background-color: white; }
    </style>
    """

    # Apply selected theme
    if theme == "Dark Modern":
        st.markdown(dark_modern, unsafe_allow_html=True)
    elif theme == "Light Modern":
        st.markdown(light_modern, unsafe_allow_html=True)
    else:
        st.markdown(light_classic, unsafe_allow_html=True)

    # Original title and description
    st.title("🤖 Enhanced RAG Chatbot with Advanced Processing")
    st.markdown("Process scientific documents, PDFs, and code with advanced equation and figure recognition")
    st.info("✨ Features: Folder processing, equation detection, figure extraction, smart caching")

    # Initialize session state
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = EnhancedRAGChatbot()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "documents_loaded" not in st.session_state:
        st.session_state.documents_loaded = False

    # Setup progress callback
    def progress_callback(message, progress=None):
        if hasattr(st.session_state, 'progress_bar') and progress is not None:
            st.session_state.progress_bar.progress(progress)
        if hasattr(st.session_state, 'status_text'):
            st.session_state.status_text.text(message)

    # Check available Ollama models
    def get_available_models():
        try:
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                models = response.json()
                return [model["name"] for model in models["models"]]
            return []
        except:
            return []

    st.session_state.chatbot.set_progress_callback(progress_callback)

    # Enhanced metrics dashboard (if documents are loaded)
    if st.session_state.documents_loaded:
        st.markdown("### 📊 Document Analysis")
        metrics_cols = st.columns(4)
        
        docs = st.session_state.chatbot.documents
        figures_count = sum(1 for doc in docs if doc.metadata.get("has_figures", False))
        equations_count = sum(1 for doc in docs if doc.metadata.get("has_equations", False))
        
        with metrics_cols[0]:
            st.metric("📄 Documents", len(docs))
        with metrics_cols[1]:
            st.metric("📊 Figures", figures_count)
        with metrics_cols[2]:
            st.metric("📐 Equations", equations_count)
        with metrics_cols[3]:
            st.metric("📚 Sources", len(set(doc.metadata.get("source") for doc in docs)))

    # Sidebar Configuration (Original + Enhanced)
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Original model configuration
        with st.expander("🔧 Model Settings", expanded=True):
            # Check available models
            available_models = get_available_models()
            
            embed_provider = st.selectbox(
                "Embedding Provider",
                ["HuggingFace", "Ollama"]
            )
            
            embedding_model = st.selectbox(
                "Embedding Model",
                ["all-MiniLM-L6-v2", "mxbai-embed-large", "nomic-embed-text"] if embed_provider == "HuggingFace"
                else available_models
            )
            
            llm_model = st.selectbox(
                "LLM Model",
                ["llama3.2", "llama3.1", "codellama", "mistral", "phi3", "qwen2.5"]
            )
            
            if st.button("🚀 Initialize Models"):
                with st.spinner("Initializing models..."):
                    if st.session_state.chatbot.initialize_models(
                        embedding_model=embedding_model,
                        llm_model=llm_model,
                        use_ollama_embeddings=(embed_provider == "Ollama")
                    ):
                        st.success("✅ Models initialized!")
                    else:
                        st.error("❌ Initialization failed")

        # Document Processing Section
        st.header("📄 Document Processing")
        
        # Folder processing (Original)
        with st.expander("📁 Process PDF Folder", expanded=True):
            folder_path = st.text_input(
                "PDF Folder Path",
                placeholder="/path/to/pdf/folder"
            )
            
            if st.button("🔄 Process PDF Folder"):
                if not folder_path:
                    st.error("Please enter a folder path")
                elif not st.session_state.chatbot.embeddings:
                    st.error("Please initialize models first")
                else:
                    try:
                        with st.spinner("Processing PDF folder..."):
                            st.session_state.progress_bar = st.progress(0)
                            st.session_state.status_text = st.empty()
                            
                            # Run async processing
                            documents = asyncio.run(
                                st.session_state.chatbot.process_pdf_folder_async(folder_path)
                            )
                            
                            if documents:
                                # Process documents
                                if st.session_state.chatbot.process_documents(documents):
                                    st.session_state.documents_loaded = True
                                    st.success(f"✅ Successfully processed {len(documents)} documents from folder!")
                                else:
                                    st.error("❌ Failed to process documents")
                            else:
                                st.warning("⚠️ No documents found in folder")
                                
                    except Exception as e:
                        st.error(f"❌ Error processing folder: {str(e)}")
        
        # File upload processing (Original)
        with st.expander("📎 Upload Files", expanded=True):
            pdf_files = st.file_uploader(
                "Upload PDF files",
                type=['pdf'],
                accept_multiple_files=True
            )
            
            code_files = st.file_uploader(
                "Upload code files",
                type=["py", "js", "java", "cpp", "txt", "md"],
                accept_multiple_files=True
            )
            
            if st.button("📚 Process Uploaded Files"):
                process_uploads(pdf_files, code_files)

        # Web Scraping Section
        with st.expander("🌐 Web Scraping", expanded=True):
            url = st.text_input(
                "Website URL",
                placeholder="https://example.com"
            )
            max_depth = st.slider(
                "Scraping Depth",
                min_value=1,
                max_value=5,
                value=2,
                help="Maximum number of links to follow from main page"
            )
            
            if st.button("🔄 Process Website"):
                if not url:
                    st.error("Please enter a URL")
                elif not st.session_state.chatbot.embeddings:
                    st.error("Please initialize models first")
                else:
                    try:
                        with st.spinner("Scraping website..."):
                            documents = st.session_state.chatbot.scrape_website(url, max_depth)
                            
                            if documents:
                                if st.session_state.chatbot.process_documents(documents):
                                    st.session_state.documents_loaded = True
                                    st.success(f"✅ Successfully processed {len(documents)} pages from website!")
                                else:
                                    st.error("❌ Failed to process documents")
                            else:
                                st.warning("⚠️ No content found to process")
                    except Exception as e:
                        st.error(f"❌ Error processing website: {str(e)}")

        # Enhanced Statistics Section
        with st.expander("📊 Advanced Statistics", expanded=True):
            if st.session_state.documents_loaded:
                show_advanced_statistics()

        # Cache Management (Original)
        with st.expander("🗄️ Cache Management"):
            show_cache_management()

    # Main Chat Interface
    if not st.session_state.documents_loaded:
        show_welcome_message()
        return

    # Enhanced Chat History Display
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(f"""
            <div class="{message['role']}-message">
                {message['content']}
            </div>
            """, unsafe_allow_html=True)
            
            # Display figures and equations
            if message.get("figures") or message.get("equations"):
                with st.expander("📎 Related Content"):
                    show_related_content(message)
            
            # Show sources
            if "sources" in message:
                with st.expander("📚 Sources"):
                    show_sources(message["sources"])
            
            # Show processing time
            if "processing_time" in message:
                st.caption(f"⏱️ Generated in {message['processing_time']:.2f}s")

    # Chat Input
    if prompt := st.chat_input("🔍 Ask about your documents..."):
        handle_user_input(prompt)

def show_sources(sources):
    """Display source documents with formatting"""
    st.markdown("#### 📚 Source Documents")
    
    for i, source in enumerate(sources, 1):
        if isinstance(source, str):
            # Simple string source
            st.markdown(f"**{i}.** {source}")
        else:
            # Document object with metadata
            metadata = source.metadata if hasattr(source, 'metadata') else {}
            source_name = metadata.get('source', 'Unknown')
            page = metadata.get('page', '')
            doc_type = metadata.get('type', 'document')
            
            source_info = f"**{i}.** {source_name}"
            if page:
                source_info += f" (Page {page})"
            if doc_type:
                source_info += f" [{doc_type}]"
                
            st.markdown(source_info)
            
            # Show additional metadata if present
            if metadata.get('has_figures') or metadata.get('has_equations'):
                extra_info = []
                if metadata.get('has_figures'):
                    extra_info.append("Contains figures")
                if metadata.get('has_equations'):
                    extra_info.append("Contains equations")
                st.markdown(f"*{' • '.join(extra_info)}*")

def show_advanced_statistics():
    """Display advanced document statistics"""
    docs = st.session_state.chatbot.documents
    
    # Document type breakdown
    doc_types = {}
    equations_by_type = {}
    figures_by_type = {}
    
    for doc in docs:
        doc_type = doc.metadata.get('type', 'unknown')
        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        if doc.metadata.get('has_equations'):
            equations_by_type[doc_type] = equations_by_type.get(doc_type, 0) + len(doc.metadata.get('equations', []))
        
        if doc.metadata.get('has_figures'):
            figures_by_type[doc_type] = figures_by_type.get(doc_type, 0) + len(doc.metadata.get('images', []))
    
    # Display statistics
    st.markdown("#### Document Types")
    for doc_type, count in doc_types.items():
        st.write(f"- {doc_type.title()}: {count}")
        if doc_type in equations_by_type:
            st.write(f"  - Equations: {equations_by_type[doc_type]}")
        if doc_type in figures_by_type:
            st.write(f"  - Figures: {figures_by_type[doc_type]}")

def show_related_content(message):
    """Display figures and equations from the message"""
    cols = st.columns(2)
    
    if message.get("figures"):
        with cols[0]:
            st.markdown("#### 📊 Figures")
            for fig in message["figures"]:
                st.image(
                    base64.b64decode(fig["data"]),
                    caption=fig.get("ocr_text", "")
                )
    
    if message.get("equations"):
        with cols[1]:
            st.markdown("#### 📐 Equations")
            for eq in message["equations"]:
                st.latex(eq)

def show_welcome_message():
    """Display welcome message and features"""
    st.info("👈 Please process documents first using the sidebar")
    st.markdown("""
    ### Quick Start Guide:
    1. **Choose Theme** - Select your preferred interface theme
    2. **Initialize Models** - Select and initialize your preferred models
    3. **Add Documents** - Either:
       - Process a folder of PDFs
       - Upload individual files
       - Mix both methods
    4. **Start Chatting** - Ask questions about your documents
    
    ### Advanced Features:
    - 🎨 **Theme Selection** - Choose between dark and light themes
    - 📊 **Enhanced Visualization** - Better display of figures and equations
    - 📁 **Folder Processing** - Process entire folders of PDFs
    - 🚀 **Multiple Model Options** - Choose from various embedding and LLM models
    - 📈 **Detailed Statistics** - Track document and processing metrics
    - 💾 **Smart Caching** - Efficient document processing
    """)

def handle_user_input(prompt):
    """Process user input and generate response"""
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        with st.spinner("🤔 Analyzing..."):
            result = st.session_state.chatbot.query(prompt)
            
            if "error" in result:
                st.error(result["error"])
            else:
                st.markdown(result["answer"])
                
                # Process figures and equations
                figures = []
                equations = []
                sources = []
                
                for doc in result.get("source_documents", []):
                    if doc.metadata.get("has_figures"):
                        figures.extend(doc.metadata["images"])
                    if doc.metadata.get("has_equations"):
                        equations.extend(doc.metadata["equations"])
                    sources.append(doc.metadata.get("source", "Unknown"))
                
                # Add to messages
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "figures": figures,
                    "equations": equations,
                    "sources": sources,
                    "processing_time": result.get("processing_time", 0)
                })
                
                # Show supporting content
                if figures or equations:
                    with st.expander("📎 View Supporting Content"):
                        show_related_content({
                            "figures": figures,
                            "equations": equations
                        })

def show_cache_management():
    """Display cache statistics and controls"""
    stats = st.session_state.chatbot.get_cache_stats()
    
    if stats:
        # Display overall stats
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Cache Files", stats['cache_files'])
        with col2:
            st.metric("Total Size", f"{stats['total_size_mb']} MB")
        
        # Display cache categories
        if 'categories' in stats:
            st.markdown("#### Cache Breakdown")
            categories_data = stats['categories']
            
            # Create columns for categories
            cols = st.columns(len(categories_data))
            for i, (category, count) in enumerate(categories_data.items()):
                with cols[i]:
                    st.metric(
                        category.title(),
                        count,
                        help=f"Number of {category} files in cache"
                    )
        
        # Cache location
        st.info(f"📁 Cache location: {stats['cache_dir']}")
        
        # Clear cache button with confirmation
        if st.button("🗑️ Clear Cache"):
            if st.session_state.chatbot.clear_cache():
                st.success("✅ Cache cleared successfully!")
                st.rerun()
            else:
                st.error("❌ Failed to clear cache")
    else:
        st.info("No cache statistics available")

def process_uploads(pdf_files, code_files):
    """Process uploaded PDF and code files"""
    if not st.session_state.chatbot.embeddings:
        st.error("Please initialize models first")
        return
    
    try:
        documents = []
        with st.spinner("Processing uploaded files..."):
            st.session_state.progress_bar = st.progress(0)
            st.session_state.status_text = st.empty()
            
            # Process PDFs if any
            if pdf_files:
                pdf_docs = st.session_state.chatbot.load_pdf_documents(pdf_files)
                documents.extend(pdf_docs)
                st.session_state.status_text.text(f"Processed {len(pdf_docs)} PDF documents")
            
            # Process code files if any
            if code_files:
                code_docs = st.session_state.chatbot.load_code_documents(code_files)
                documents.extend(code_docs)
                st.session_state.status_text.text(
                    f"Processed {len(documents)} documents ({len(pdf_docs) if pdf_files else 0} PDFs, {len(code_docs) if code_files else 0} code files)"
                )
            
            if documents:
                if st.session_state.chatbot.process_documents(documents):
                    st.session_state.documents_loaded = True
                    st.success(f"✅ Successfully processed {len(documents)} documents!")
                else:
                    st.error("❌ Failed to process documents")
            else:
                st.warning("⚠️ No documents to process")
                
    except Exception as e:
        st.error(f"❌ Error processing files: {str(e)}")

if __name__ == "__main__":
    main()