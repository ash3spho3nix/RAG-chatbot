import pytest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture
def test_docs_path():
    return os.path.join("tests", "test_data", "docs")

@pytest.fixture
def vector_store_path():
    return os.path.join("tests", "test_data", "vector_store")