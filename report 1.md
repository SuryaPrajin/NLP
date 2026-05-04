# Project Analysis Report: Legal AI Assistant (Antigravity-Legal)

This report provides a comprehensive analysis of the code and file structure of the **Legal AI Assistant** project. Every component, from the core machine learning models to the web infrastructure, is detailed below.

---

## 1. Project Overview
The project is a domain-specific, "law-aware" AI assistant designed to provide grounded legal information based on Indian law (IPC, CrPC, IT Act, etc.). It uses a **Retrieval-Augmented Generation (RAG)** architecture combined with a **Custom Transformer LLM** built from scratch.

---

## 2. File Structure & Directory Analysis

### Root Directory
- **`.env`**: Contains sensitive environment variables (e.g., API keys, configuration settings). Used to keep secrets out of the source code.
- **`training_log.txt`**: Records the progress of the model training process, including loss metrics and timestamps.
- **`report 1.md`**: This analysis report.

### `backend/` (Core Logic)
The backend is built with **FastAPI**, a high-performance Python web framework.
- **`main.py`**: The entry point for the FastAPI server.
    - **What**: Sets up CORS, rate limiting, and includes API routes.
    - **Why**: Handles incoming requests, provides admin statistics, and manages the application lifecycle.
- **`requirements.txt`**: Lists all Python dependencies (e.g., `fastapi`, `torch`, `faiss-cpu`, `sentence-transformers`).
- **`.env`**: Local environment settings for the backend.

#### `backend/core/` (Intelligence Layer)
This is where the "brain" of the assistant lives.
- **`custom_llm.py`**:
    - **What**: A from-scratch implementation of a Transformer architecture using **PyTorch**.
    - **Why**: Allows for a fully sovereign, offline-capable legal reasoning engine. It implements Multi-Head Attention, Feed-Forward layers, and Positional Encoding.
- **`bpe_tokenizer.py`**:
    - **What**: A custom Byte Pair Encoding (BPE) tokenizer.
    - **Why**: Efficiently compresses legal terminology into tokens, helping the model understand "concepts" rather than just "characters".
- **`rag_engine.py`**:
    - **What**: A hybrid search engine using **FAISS** (vector search) and **BM25** (keyword search).
    - **Why**: Combines the semantic understanding of vectors with the exact matching of keywords to retrieve the most relevant legal sections. It also uses a **Cross-Encoder** for precise re-ranking of results.
- **`intent_classifier.py`**:
    - **What**: A rule-based and logic-driven classifier.
    - **Why**: Identifies if a user is asking for a definition, punishment, procedure, or rights, allowing the system to tailor its response strategy.
- **`memory_store.py`**:
    - **What**: Manages session history and user profiles.
    - **Why**: Enables multi-turn conversations by keeping track of what was previously discussed.
- **`agent_logic.py`**:
    - **What**: A state machine that decides the high-level response strategy (e.g., greeting, escalation, or LLM generation).
- **`dataset_loader.py` & `trainer.py`**:
    - **What**: Handles data ingestion from PDFs and manages the Supervised Fine-Tuning (SFT) loop.
    - **Why**: Allows the model to learn specifically from the provided legal documents.
- **`instruction_gen.py`**:
    - **What**: Automatically generates Question-Answer pairs from legal texts.
    - **Why**: Creates synthetic training data to fine-tune the LLM on legal reasoning.
- **`logger.py`**:
    - **What**: An audit logger for tracking user interactions.
    - **Why**: Essential for monitoring system performance and ensuring accountability.

#### `backend/services/` (Infrastructure Layer)
- **`llm_service.py`**:
    - **What**: A wrapper around the `CustomLegalLLM`.
    - **Why**: Implements a **Circuit Breaker** pattern to handle high load or failures gracefully.
- **`embedding_service.py`**:
    - **What**: Generates vector embeddings for text using Sentence-Transformers.
    - **Why**: Necessary for the semantic search component of the RAG engine.
- **`action_router.py`**:
    - **What**: Routes specific intents to external actions (like creating a support ticket).

#### `backend/routes/` (API Layer)
- **`chat.py`**:
    - **What**: Defines the `/api/v1/chat` endpoint.
    - **Why**: Orchestrates the entire flow: Intent classification -> Query expansion -> RAG retrieval -> LLM generation -> Verification -> Logging.

#### `backend/db/` (Data Store)
- **`policies.json`**: Contains static business policies or fallback guidelines.

### `frontend/` (User Interface)
The frontend is a modern web application built with **React** and **Vite**.
- **`src/App.jsx`**: The main UI component.
    - **What**: Implements the chat interface with support for loading states, markdown rendering, and error handling.
    - **Why**: Provides a premium, interactive experience for the user.
- **`src/index.css` & `App.css`**:
    - **What**: Custom Vanilla CSS for styling.
    - **Why**: Ensures a sleek, "legal-professional" aesthetic with glassmorphism and smooth transitions.
- **`package.json`**: Defines the frontend dependencies and scripts.

### `ML/` (Knowledge Base)
This directory contains the "Source of Truth" for the AI.
- **`Criminal Law/`**: Contains PDF files of major Indian Acts:
    - `Indian Penal Code.pdf`
    - `The-Indian-Evidence-Act-1872.pdf`
    - `the_code_of_criminal_procedure,_1973.pdf`
    - `it_act_2000_updated.pdf`
    - `case_laws_database.pdf`
- **Why**: These are processed by the `rag_engine` to provide grounded, verifiable legal citations.

---

## 3. Technology Stack Breakdown

| Component | Technology | Rationale |
| :--- | :--- | :--- |
| **Backend Framework** | FastAPI | High speed, async support, and automatic documentation. |
| **Deep Learning** | PyTorch | Flexibility for building custom Transformer architectures. |
| **Vector Search** | FAISS | Industry-standard for fast similarity search in large datasets. |
| **Keyword Search** | BM25 | Provides robust exact-match retrieval where semantic search might miss specific terms. |
| **Embeddings** | Sentence-Transformers | High-quality pre-trained models for semantic understanding. |
| **Frontend** | React + Vite | Fast development, component-based architecture, and excellent performance. |
| **Tokenization** | Custom BPE | Optimized for the specific vocabulary of legal texts. |

---

## 4. Key Workflow: The "Life of a Query"
1. **User asks**: "What is the punishment for theft?"
2. **Intent Classification**: System identifies `PUNISHMENT` and `LEGAL_DEFINITION`.
3. **Query Expansion**: The query is expanded to include terms like "penalty", "jail", "Section 378", "Section 379".
4. **Hybrid Retrieval**:
    - **FAISS** finds semantically similar sections.
    - **BM25** finds sections containing the word "theft".
5. **Re-ranking**: A Cross-Encoder ranks the combined results, prioritizing exact Act/Section matches.
6. **LLM Generation**: The `CustomLegalLLM` synthesizes an answer using the retrieved context, ensuring it includes a relevant law citation and an explanation.
7. **Verification**: The system checks if the model's citation matches the retrieved documents.
8. **UI Update**: The user receives a structured response with a mandatory legal disclaimer.

---

## 5. Security & Safety Features
- **Rate Limiting**: Prevents API abuse using `slowapi`.
- **Circuit Breaker**: Protects the system from cascading failures if the LLM is overloaded.
- **Audit Logging**: Every interaction is logged for quality assurance and debugging.
- **Grounded Responses**: The system is strictly instructed (via RAG) to only answer based on provided law books to prevent hallucinations.
- **Legal Disclaimer**: Every response includes a clear warning that the AI is for informational purposes only.

---
**Report generated by Antigravity AI.**
