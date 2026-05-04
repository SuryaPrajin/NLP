import asyncio
import os
import sys

# Add project root to sys.path
sys.path.append(os.getcwd())

from backend.core.rag_engine import rag_engine
from backend.services.llm_service import llm_service
from backend.core.response_validator import response_validator

async def test_legal_flow():
    query = "What is the punishment for murder under IPC?"
    print(f"Testing Query: {query}")
    
    # 1. Test RAG
    print("\n--- RAG Retrieval ---")
    context, score = rag_engine.retrieve(query)
    print(f"Similarity Score: {score}")
    print(f"Context Length: {len(context)}")
    print(f"Context Snippet: {context[:200]}...")
    
    # 2. Test LLM Service
    print("\n--- LLM Response ---")
    response = await llm_service.generate_response(query, [], context, {}, ["PUNISHMENT"])
    print(f"Response Keys: {list(response.keys())}")
    print(f"Response Message: {response.get('message')}")
    
    # 3. Test Validator
    print("\n--- Validator Check ---")
    is_valid = response_validator.validate(response, query)
    print(f"Is Valid: {is_valid}")
    
    if is_valid:
        print("\nSUCCESS: The flow is working correctly.")
    else:
        print("\nFAILURE: Validation failed.")

if __name__ == "__main__":
    asyncio.run(test_legal_flow())
