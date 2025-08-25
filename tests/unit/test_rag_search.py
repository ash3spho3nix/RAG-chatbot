import pytest
from src.v1.RAG_Search_new import create_vector_store_from_pdfs, RAG_search, load_vector_store

def test_vector_store_creation(test_docs_path, vector_store_path):
    vector_store = create_vector_store_from_pdfs(test_docs_path)
    assert vector_store is not None
    assert hasattr(vector_store, 'similarity_search')

def test_rag_search():
    query = "test query"
    response = RAG_search(query)
    assert isinstance(response, str)
    assert len(response) > 0

def test_vector_store_loading(vector_store_path):
    vector_store = load_vector_store(vector_store_path)
    assert vector_store is not None