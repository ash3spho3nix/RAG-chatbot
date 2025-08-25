import pytest
from src.v1.response_functions import optimize_response

def test_response_optimization():
    test_response = "Test response text"
    optimized = optimize_response(test_response)
    assert isinstance(optimized, str)
    assert len(optimized) > 0