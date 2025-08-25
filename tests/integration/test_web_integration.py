import pytest
from fastapi.testclient import TestClient
from src.v1.RAG_Search_web import app

client = TestClient(app)

def test_search_endpoint():
    response = client.post(
        "/search",
        json={"query": "test question"}
    )
    assert response.status_code == 200
    assert "response" in response.json()

def test_upload_endpoint():
    with open("tests/test_data/docs/test.pdf", "rb") as f:
        response = client.post(
            "/upload",
            files={"file": ("test.pdf", f, "application/pdf")}
        )
        assert response.status_code == 200