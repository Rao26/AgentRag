import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration settings for the RAG system"""
    
    # Cerebras API configuration (OpenAI-compatible endpoint)
    CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
    
    # Cerebras API endpoint (OpenAI-compatible)
    CEREBRAS_API_BASE = "https://api.cerebras.ai/v1"
    
    # Model configuration
    MODEL_NAME = "gpt-oss-120b"
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
    
    # Vector store configuration
    CHROMA_PERSIST_DIR = "./chroma_db"
    CHROMA_COLLECTION_NAME = "document_embeddings"
    
    # Text processing configuration
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Agent configuration
    MAX_CONTEXT_LENGTH = 4000
    TEMPERATURE = 0.1
    
    @classmethod
    def validate_config(cls):
        """Validate that all required configuration is present"""
        if not cls.CEREBRAS_API_KEY:
            raise ValueError(
                "CEREBRAS_API_KEY environment variable is required.\n"
                "Please create a .env file in the project root with:\n"
                "CEREBRAS_API_KEY=your_api_key_here\n\n"
                "You can get your API key from: https://www.cerebras.net/"
            )

config = Config()