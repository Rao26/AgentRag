"""
Agentic RAG Document Search System

A Proof of Concept for intelligent document search using agentic workflows.
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"
__description__ = "Agentic RAG: AI Assistant for Document Search using Agents"

from .config import config
from .vectorstore import VectorStoreManager
from .agent import DocumentSearchAgent
from .ui import StreamlitUI

__all__ = [
    'config',
    'VectorStoreManager', 
    'DocumentSearchAgent',
    'StreamlitUI'
]