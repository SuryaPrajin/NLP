import asyncio
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.services.llm_service import llm_service
from backend.services.embedding_service import embedding_service

async def test_custom_system():
    print("--- Testing Embedding Service ---")
    text = "Section 302 of IPC refers to murder."
    emb = embedding_service.get_embedding(text)
    print(f"Embedding generated. Shape: {emb.shape}")
    
    print("\n--- Testing Custom LLM Service ---")
    prompt = "What is the punishment for theft?"
    # Simulated RAG context
    context = '[{"section": "Section 378", "act": "IPC", "title": "Theft", "content": "Whoever, intending to take dishonestly any movable property out of the possession of any person without that person\'s consent, moves that property in order to such taking, is said to commit theft."}]'
    
    response = await llm_service.generate_response(prompt, [], context)
    print("Generated Response:")
    import json
    print(json.dumps(response, indent=2))

if __name__ == "__main__":
    asyncio.run(test_custom_system())
