"""
RAG Document Search System

A standard Retrieval-Augmented Generation system for document search.
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"
__description__ = "RAG: Document Search using Retrieval-Augmented Generation"

from .config import config
from .vectorstore import VectorStoreManager
from .rag_engine import RAGEngine
from .ui import StreamlitUI

__all__ = [
    'config',
    'VectorStoreManager', 
    'RAGEngine',
    'StreamlitUI'
]
