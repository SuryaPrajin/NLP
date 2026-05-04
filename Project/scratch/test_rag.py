import sys
import os
# Add the project root to sys.path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.core.rag_engine import rag_engine

def test_rag():
    query = "What is punishment for murder in IPC?"
    print(f"Query: {query}")
    context = rag_engine.retrieve(query, k=2)
    print("\nRetrieved Context:")
    print("-" * 50)
    print(context)
    print("-" * 50)

if __name__ == "__main__":
    # Ensure dependencies are available
    try:
        test_rag()
    except Exception as e:
        print(f"Error: {e}")
