# music-insights-llm

This is a personal project in which I design an agentic RAG system that analyzes personal Spotify listening data to answer complex questions about listening behavior, trends, and preferences.

The system combines: 
* local llms (ollama)
* vector search (faiss/chroma)
* structured data querying (python tools)
* agent-based orchestration (langchain/langgraph)?


## key features

Agentic Query Routing  
(working on it..)

RAG Pipeline  
(...)

Tool-calling System  
(...)

Local LLM Integration  
(...)

Evaluation Framework  
(...)

## architecture
```
User Query
    ↓
Agent (LangChain / LangGraph)
    ↓
Decision:
  ├── Tool Call (structured Python functions)
  └── RAG Pipeline (vector retrieval)
    ↓
LLM (Ollama)
    ↓
Final Response
```
## example queries
"how has my music tate changed over time?"  
"what genres did I listen to the most in 2023?"  

## evaluation 

## future plans
work on adding a real-time spotify api intergration  
