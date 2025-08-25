"""
Response optimization functions for Enhanced RAG Chatbot
Provides faster embedding, vector search, LLM streaming, and vectorstore persistence
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator, Tuple
import logging
from datetime import datetime
import hashlib
import time

import faiss
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
import torch
torch.classes.__path__ = []
# Setup logger
logger = logging.getLogger(__name__)

class OptimizedEmbeddings:
    """Optimized embedding class with CPU acceleration and caching"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_size: int = 1000):
        """
        Initialize optimized embeddings
        
        Args:
            model_name: Fast, lightweight embedding model
            cache_size: Number of embeddings to cache in memory
        """
        self.model_name = model_name
        self.cache_size = cache_size
        self.embedding_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Use HuggingFaceEmbeddings instead of SentenceTransformer directly
        self.model = HuggingFaceEmbeddings(
            model_name=f"sentence-transformers/{model_name}",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True},
            cache_folder='./model_cache'  # Local cache directory
        )
        
        # Enable CPU optimizations
        if hasattr(torch, 'set_num_threads'):
            torch.set_num_threads(os.cpu_count())
        
        logger.info(f"Initialized optimized embeddings with {model_name}")
    
    def embed_query(self, text: str) -> List[float]:
        """Embed single query with caching"""
        # Create cache key
        cache_key = hashlib.md5(text.encode()).hexdigest()
        
        # Check cache first
        if cache_key in self.embedding_cache:
            self.cache_hits += 1
            return self.embedding_cache[cache_key]
        
        # Generate embedding using HuggingFaceEmbeddings
        embedding = self.model.embed_query(text)
        
        # Cache the result
        self._add_to_cache(cache_key, embedding)
        self.cache_misses += 1
        
        return embedding
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents with batch processing and caching"""
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            cache_key = hashlib.md5(text.encode()).hexdigest()
            if cache_key in self.embedding_cache:
                embeddings.append(self.embedding_cache[cache_key])
                self.cache_hits += 1
            else:
                embeddings.append(None)  # Placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)
                self.cache_misses += 1
        
        # Batch process uncached texts
        if uncached_texts:
            batch_embeddings = self.model.embed_documents(uncached_texts)
            
            # Cache and insert results
            for idx, embedding in zip(uncached_indices, batch_embeddings):
                embeddings[idx] = embedding
                
                # Cache the result
                cache_key = hashlib.md5(texts[idx].encode()).hexdigest()
                self._add_to_cache(cache_key, embedding)
        
        return embeddings
    
    def _add_to_cache(self, key: str, embedding: List[float]):
        """Add embedding to cache with size management"""
        if len(self.embedding_cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.embedding_cache))
            del self.embedding_cache[oldest_key]
        
        self.embedding_cache[key] = embedding
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total * 100) if total > 0 else 0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate_percent': round(hit_rate, 2),
            'cache_size': len(self.embedding_cache),
            'max_cache_size': self.cache_size
        }

class OptimizedVectorStore:
    """Optimized FAISS vector store with advanced indexing"""
    
    def __init__(self, embedding_dim: int, use_gpu: bool = False):
        """
        Initialize optimized vector store
        
        Args:
            embedding_dim: Dimension of embeddings
            use_gpu: Whether to use GPU acceleration
        """
        self.embedding_dim = embedding_dim
        self.use_gpu = use_gpu
        self.documents = []
        self.metadata = []
        
        # Choose optimal index type based on expected size
        self.index_type = "flat"  # Will be updated based on document count
        self.index = None
        self.is_trained = False
        
        logger.info(f"Initialized optimized vector store (dim={embedding_dim})")
    
    def _create_optimal_index(self, num_vectors: int) -> faiss.Index:
        """Create optimal FAISS index based on dataset size"""
        
        if num_vectors < 1000:
            # Small dataset: use flat index (exact search)
            index = faiss.IndexFlatIP(self.embedding_dim)
            self.index_type = "flat"
            
        elif num_vectors < 10000:
            # Medium dataset: use IVF with flat quantizer
            nlist = min(100, num_vectors // 10)
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
            self.index_type = "ivf_flat"
            
        else:
            # Large dataset: use IVF with product quantization
            nlist = min(1000, num_vectors // 50)
            m = 8  # Number of subquantizers
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            index = faiss.IndexIVFPQ(quantizer, self.embedding_dim, nlist, m, 8)
            self.index_type = "ivf_pq"
        
        # GPU acceleration if available and requested
        if self.use_gpu and faiss.get_num_gpus() > 0:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                logger.info("Enabled GPU acceleration for FAISS index")
            except Exception as e:
                logger.warning(f"GPU acceleration failed: {e}")
        
        logger.info(f"Created {self.index_type} index for {num_vectors} vectors")
        return index
    
    def add_documents(self, documents: List[Document], embeddings: np.ndarray):
        """Add documents and embeddings to the index"""
        if self.index is None or len(documents) != len(embeddings):
            self.index = self._create_optimal_index(len(embeddings))
        
        # Train index if necessary
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            logger.info("Training FAISS index...")
            self.index.train(embeddings.astype('float32'))
            self.is_trained = True
        
        # Add vectors to index
        embeddings_float32 = embeddings.astype('float32')
        self.index.add(embeddings_float32)
        
        # Store documents and metadata
        self.documents.extend(documents)
        for i, doc in enumerate(documents):
            metadata = doc.metadata.copy()
            metadata['doc_id'] = len(self.metadata) + i
            self.metadata.append(metadata)
        
        logger.info(f"Added {len(documents)} documents to index")
    
    def similarity_search(self, query_embedding: List[float], k: int = 5, 
                         search_params: Optional[Dict] = None) -> List[Tuple[Document, float]]:
        """Optimized similarity search with configurable parameters"""
        if self.index is None:
            return []
        
        # Prepare query
        query_vector = np.array([query_embedding]).astype('float32')
        
        # Set search parameters for IVF indices
        if self.index_type.startswith('ivf') and search_params:
            nprobe = search_params.get('nprobe', 10)
            if hasattr(self.index, 'nprobe'):
                self.index.nprobe = nprobe
        
        # Perform search
        start_time = time.time()
        scores, indices = self.index.search(query_vector, k)
        search_time = time.time() - start_time
        
        # Prepare results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.documents):
                results.append((self.documents[idx], float(score)))
        
        logger.debug(f"Vector search completed in {search_time:.3f}s")
        return results
    
    def get_index_info(self) -> Dict[str, Any]:
        """Get information about the current index"""
        if self.index is None:
            return {"status": "not_initialized"}
        
        info = {
            "index_type": self.index_type,
            "num_vectors": self.index.ntotal,
            "dimension": self.embedding_dim,
            "is_trained": getattr(self.index, 'is_trained', True),
            "memory_usage_mb": self.index.ntotal * self.embedding_dim * 4 / (1024 * 1024)
        }
        
        # Add index-specific info
        if hasattr(self.index, 'nlist'):
            info['nlist'] = self.index.nlist
        if hasattr(self.index, 'nprobe'):
            info['nprobe'] = self.index.nprobe
            
        return info

class StreamingLLMResponse:
    """Streaming LLM responses with context optimization"""
    
    def __init__(self, llm: OllamaLLM, max_context_length: int = 4000):
        """
        Initialize streaming LLM response handler
        
        Args:
            llm: Ollama LLM instance
            max_context_length: Maximum context length to send to LLM
        """
        self.llm = llm
        self.max_context_length = max_context_length
        self.response_cache = {}
        
        logger.info("Initialized streaming LLM response handler")
    
    def optimize_context(self, query: str, retrieved_docs: List[Document]) -> str:
        """Optimize context by selecting most relevant parts and truncating if needed"""
        if not retrieved_docs:
            return "No relevant documents found."
        
        # Score documents by relevance (simple keyword matching for now)
        query_words = set(query.lower().split())
        scored_docs = []
        
        for doc in retrieved_docs:
            content = doc.page_content.lower()
            relevance_score = sum(1 for word in query_words if word in content)
            scored_docs.append((doc, relevance_score))
        
        # Sort by relevance score
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Build context string within length limit
        context_parts = []
        current_length = 0
        
        for doc, score in scored_docs:
            content = doc.page_content
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', '')
            
            doc_text = f"Source: {source}"
            if page:
                doc_text += f" (Page {page})"
            doc_text += f"\nContent: {content}\n\n"
            
            if current_length + len(doc_text) > self.max_context_length:
                # Truncate the last document if needed
                remaining_space = self.max_context_length - current_length - 100
                if remaining_space > 200:  # Only add if meaningful space remains
                    truncated_content = content[:remaining_space] + "..."
                    doc_text = f"Source: {source}"
                    if page:
                        doc_text += f" (Page {page})"
                    doc_text += f"\nContent: {truncated_content}\n\n"
                    context_parts.append(doc_text)
                break
            
            context_parts.append(doc_text)
            current_length += len(doc_text)
        
        return "".join(context_parts)
    
    def generate_streaming_response(self, query: str, context: str) -> Iterator[str]:
        """Generate streaming response from LLM"""
        # Create optimized prompt
        prompt = f"""Based on the following context, answer the question. Be concise and accurate.

Context:
{context}

Question: {query}

Answer:"""
        
        # Check cache first
        cache_key = hashlib.md5((query + context).encode()).hexdigest()
        if cache_key in self.response_cache:
            # Simulate streaming for cached response
            cached_response = self.response_cache[cache_key]
            words = cached_response.split()
            for i in range(0, len(words), 3):  # Stream 3 words at a time
                yield " ".join(words[i:i+3]) + " "
            return
        
        try:
            # Stream response from LLM
            response_text = ""
            
            # Note: Ollama doesn't directly support streaming in langchain
            # This is a placeholder for the streaming implementation
            # In practice, you'd use the Ollama API directly for streaming
            
            full_response = self.llm.invoke(prompt)
            
            # Simulate streaming by yielding chunks
            words = full_response.split()
            for i in range(0, len(words), 2):
                chunk = " ".join(words[i:i+2]) + " "
                response_text += chunk
                yield chunk
            
            # Cache the complete response
            self.response_cache[cache_key] = full_response
            
        except Exception as e:
            logger.error(f"Error generating streaming response: {e}")
            yield f"Error generating response: {str(e)}"
    
    def generate_response(self, query: str, context: str) -> str:
        """Generate complete response (non-streaming)"""
        # Check cache first
        cache_key = hashlib.md5((query + context).encode()).hexdigest()
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]
        
        prompt = f"""Based on the following context, answer the question. Be concise and accurate.

Context:
{context}

Question: {query}

Answer:"""
        
        try:
            response = self.llm.invoke(prompt)
            self.response_cache[cache_key] = response
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"

class VectorStoreManager:
    """Manager for saving and loading vectorstores with metadata"""
    
    def __init__(self, base_dir: str = "./vectorstores"):
        """
        Initialize vectorstore manager
        
        Args:
            base_dir: Base directory for storing vectorstores
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        logger.info(f"VectorStore manager initialized at {base_dir}")
    
    def save_vectorstore(self, vectorstore: FAISS, documents: List[Document], 
                        name: str, metadata: Optional[Dict] = None) -> bool:
        """
        Save complete vectorstore with documents and metadata
        
        Args:
            vectorstore: FAISS vectorstore instance
            documents: Original documents
            name: Name for the saved vectorstore
            metadata: Additional metadata to save
        
        Returns:
            bool: Success status
        """
        try:
            # Create directory for this vectorstore
            vs_dir = self.base_dir / name
            vs_dir.mkdir(exist_ok=True)
            
            # Save FAISS index
            vectorstore.save_local(str(vs_dir / "faiss_index"))
            
            # Save documents
            with open(vs_dir / "documents.pkl", 'wb') as f:
                pickle.dump(documents, f)
            
            # Prepare and save metadata
            save_metadata = {
                "name": name,
                "created_at": datetime.now().isoformat(),
                "num_documents": len(documents),
                "embedding_dimension": vectorstore.index.d if hasattr(vectorstore.index, 'd') else None,
                "index_type": type(vectorstore.index).__name__,
                "total_vectors": vectorstore.index.ntotal if hasattr(vectorstore.index, 'ntotal') else 0
            }
            
            if metadata:
                save_metadata.update(metadata)
            
            with open(vs_dir / "metadata.json", 'w') as f:
                json.dump(save_metadata, f, indent=2)
            
            # Save document statistics
            doc_stats = self._calculate_doc_stats(documents)
            with open(vs_dir / "document_stats.json", 'w') as f:
                json.dump(doc_stats, f, indent=2)
            
            logger.info(f"Vectorstore '{name}' saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving vectorstore '{name}': {e}")
            return False
    
    def load_vectorstore(self, name: str, embeddings) -> Optional[Tuple[FAISS, List[Document], Dict]]:
        """
        Load complete vectorstore with documents and metadata
        
        Args:
            name: Name of the vectorstore to load
            embeddings: Embedding function for loading FAISS
        
        Returns:
            Tuple of (vectorstore, documents, metadata) or None if failed
        """
        try:
            vs_dir = self.base_dir / name
            
            if not vs_dir.exists():
                logger.error(f"Vectorstore '{name}' not found")
                return None
            
            # Load FAISS index
            vectorstore = FAISS.load_local(
                str(vs_dir / "faiss_index"),
                embeddings,
                allow_dangerous_deserialization=True
            )
            
            # Load documents
            with open(vs_dir / "documents.pkl", 'rb') as f:
                documents = pickle.load(f)
            
            # Load metadata
            with open(vs_dir / "metadata.json", 'r') as f:
                metadata = json.load(f)
            
            logger.info(f"Vectorstore '{name}' loaded successfully")
            return vectorstore, documents, metadata
            
        except Exception as e:
            logger.error(f"Error loading vectorstore '{name}': {e}")
            return None
    
    def list_vectorstores(self) -> List[Dict[str, Any]]:
        """List all available vectorstores with their metadata"""
        vectorstores = []
        
        for vs_dir in self.base_dir.iterdir():
            if vs_dir.is_dir():
                metadata_file = vs_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        
                        # Add size information
                        size_mb = sum(f.stat().st_size for f in vs_dir.rglob('*') if f.is_file()) / (1024 * 1024)
                        metadata['size_mb'] = round(size_mb, 2)
                        
                        vectorstores.append(metadata)
                    except Exception as e:
                        logger.warning(f"Error reading metadata for {vs_dir.name}: {e}")
        
        return sorted(vectorstores, key=lambda x: x.get('created_at', ''), reverse=True)
    
    def delete_vectorstore(self, name: str) -> bool:
        """Delete a vectorstore and all its files"""
        try:
            vs_dir = self.base_dir / name
            if vs_dir.exists():
                import shutil
                shutil.rmtree(vs_dir)
                logger.info(f"Vectorstore '{name}' deleted successfully")
                return True
            else:
                logger.warning(f"Vectorstore '{name}' not found")
                return False
        except Exception as e:
            logger.error(f"Error deleting vectorstore '{name}': {e}")
            return False
    
    def _calculate_doc_stats(self, documents: List[Document]) -> Dict[str, Any]:
        """Calculate statistics about the documents"""
        if not documents:
            return {}
        
        doc_types = {}
        sources = set()
        total_chars = 0
        
        for doc in documents:
            # Count document types
            doc_type = doc.metadata.get('type', 'unknown')
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            
            # Collect sources
            source = doc.metadata.get('source', 'Unknown')
            sources.add(source)
            
            # Count characters
            total_chars += len(doc.page_content)
        
        return {
            'total_documents': len(documents),
            'unique_sources': len(sources),
            'document_types': doc_types,
            'total_characters': total_chars,
            'average_doc_length': total_chars // len(documents) if documents else 0
        }

class ResponseOptimizer:
    """Main class that combines all optimizations"""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", 
                 vectorstore_dir: str = "./vectorstores",
                 cache_size: int = 1000):
        """
        Initialize the response optimizer with all components
        
        Args:
            embedding_model: Name of the embedding model to use
            vectorstore_dir: Directory for vectorstore persistence
            cache_size: Size of embedding cache
        """
        # Initialize components
        self.embeddings = OptimizedEmbeddings(embedding_model, cache_size)
        self.vectorstore = None
        self.streaming_llm = None
        self.vs_manager = VectorStoreManager(vectorstore_dir)
        
        # Performance tracking
        self.query_times = []
        self.embedding_times = []
        
        logger.info("Response optimizer initialized with all components")
    
    def initialize_llm(self, llm: OllamaLLM, max_context_length: int = 4000):
        """Initialize the streaming LLM component"""
        self.streaming_llm = StreamingLLMResponse(llm, max_context_length)
    
    def create_vectorstore(self, documents: List[Document]) -> bool:
        """Create optimized vectorstore from documents"""
        try:
            start_time = time.time()
            
            # Generate embeddings
            texts = [doc.page_content for doc in documents]
            embeddings_list = self.embeddings.embed_documents(texts)
            embeddings_array = np.array(embeddings_list)
            
            embedding_time = time.time() - start_time
            self.embedding_times.append(embedding_time)
            
            # Create optimized vector store
            self.vectorstore = OptimizedVectorStore(
                embedding_dim=len(embeddings_list[0]),
                use_gpu=False  # Set to True if GPU available
            )
            
            self.vectorstore.add_documents(documents, embeddings_array)
            
            logger.info(f"Vectorstore created in {time.time() - start_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Error creating vectorstore: {e}")
            return False
    
    def query_optimized(self, query: str, k: int = 5, 
                       search_params: Optional[Dict] = None) -> Dict[str, Any]:
        """Perform optimized query with all enhancements"""
        start_time = time.time()
        
        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Perform optimized vector search
            if self.vectorstore:
                results = self.vectorstore.similarity_search(
                    query_embedding, k=k, search_params=search_params
                )
                
                # Extract documents for context
                retrieved_docs = [doc for doc, score in results]
                
                # Generate optimized response
                if self.streaming_llm:
                    context = self.streaming_llm.optimize_context(query, retrieved_docs)
                    response = self.streaming_llm.generate_response(query, context)
                else:
                    context = "\n\n".join([doc.page_content for doc in retrieved_docs[:3]])
                    response = f"Based on the context: {context[:500]}..."
                
                query_time = time.time() - start_time
                self.query_times.append(query_time)
                
                return {
                    "answer": response,
                    "source_documents": retrieved_docs,
                    "query_time": query_time,
                    "context_length": len(context),
                    "cache_stats": self.embeddings.get_cache_stats()
                }
            else:
                return {"error": "No vectorstore available"}
                
        except Exception as e:
            logger.error(f"Error in optimized query: {e}")
            return {"error": str(e)}
    
    def stream_response(self, query: str, k: int = 5) -> Iterator[str]:
        """Stream optimized response"""
        if not self.vectorstore or not self.streaming_llm:
            yield "Error: Components not initialized"
            return
        
        try:
            # Get context
            query_embedding = self.embeddings.embed_query(query)
            results = self.vectorstore.similarity_search(query_embedding, k=k)
            retrieved_docs = [doc for doc, score in results]
            context = self.streaming_llm.optimize_context(query, retrieved_docs)
            
            # Stream response
            for chunk in self.streaming_llm.generate_streaming_response(query, context):
                yield chunk
                
        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            yield f"Error: {str(e)}"
    
    def save_current_vectorstore(self, name: str, metadata: Optional[Dict] = None) -> bool:
        """Save current vectorstore"""
        if not self.vectorstore:
            logger.error("No vectorstore to save")
            return False
        
        # Convert OptimizedVectorStore to FAISS for saving
        try:
            # Create FAISS vectorstore from our optimized version
            faiss_vs = FAISS(
                embedding_function=self.embeddings.embed_query,
                index=self.vectorstore.index,
                docstore={i: doc for i, doc in enumerate(self.vectorstore.documents)},
                index_to_docstore_id={i: i for i in range(len(self.vectorstore.documents))}
            )
            
            return self.vs_manager.save_vectorstore(
                faiss_vs, self.vectorstore.documents, name, metadata
            )
        except Exception as e:
            logger.error(f"Error saving vectorstore: {e}")
            return False
    
    def load_vectorstore(self, name: str) -> bool:
        """Load saved vectorstore"""
        try:
            result = self.vs_manager.load_vectorstore(name, self.embeddings)
            if result:
                faiss_vs, documents, metadata = result
                
                # Convert back to OptimizedVectorStore
                self.vectorstore = OptimizedVectorStore(
                    embedding_dim=faiss_vs.index.d,
                    use_gpu=False
                )
                self.vectorstore.index = faiss_vs.index
                self.vectorstore.documents = documents
                self.vectorstore.metadata = [doc.metadata for doc in documents]
                
                logger.info(f"Loaded vectorstore '{name}' with {len(documents)} documents")
                return True
            return False
        except Exception as e:
            logger.error(f"Error loading vectorstore: {e}")
            return False
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        stats = {
            'embedding_cache': self.embeddings.get_cache_stats(),
            'average_query_time': np.mean(self.query_times) if self.query_times else 0,
            'average_embedding_time': np.mean(self.embedding_times) if self.embedding_times else 0,
            'total_queries': len(self.query_times),
            'vectorstore_info': self.vectorstore.get_index_info() if self.vectorstore else {}
        }
        
        return stats