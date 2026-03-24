# music-insights-llm

This is a personal project in which I design an agentic RAG system that analyzes personal Spotify listening data to answer complex questions about listening behavior, trends, and preferences.

The system combines: 
* local llms (ollama)
* vector search (faiss)
* structured data querying (python tools)
* agent-based orchestration (langchain)

## architecture
```
User Query
    ↓
Agent (LangChain / might do LangGraph instead)
    ↓
Decision:
  ├── Tool Call (structured Python functions)
  └── RAG Pipeline (vector retrieval)
    ↓
LLM (Ollama)
    ↓
Final Response
```

## rebuild vectorstore

python src/temp.py

## example queries
"how has my music tate changed over time?"  

## evaluation 
...

## future plans
work on adding a real-time spotify api intergration
use langgraph?

