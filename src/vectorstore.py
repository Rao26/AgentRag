import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

from .config import config

class VectorStoreManager:
    """Manages document loading, embedding, and vector storage"""
    
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
        )
        self.vector_store = None
    
    def load_documents(self, data_dir: str = "./data") -> List[Document]:
        """Load documents from data directory"""
        documents = []
        
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print(f"Created data directory at {data_dir}")
            return documents
        
        for filename in os.listdir(data_dir):
            file_path = os.path.join(data_dir, filename)
            try:
                if filename.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    # Add source metadata
                    for doc in docs:
                        doc.metadata['source'] = filename
                    documents.extend(docs)
                    print(f"Loaded PDF: {filename} ({len(docs)} pages)")
                elif filename.endswith(".txt"):
                    loader = TextLoader(file_path, encoding='utf-8')
                    docs = loader.load()
                    # Add source metadata
                    for doc in docs:
                        doc.metadata['source'] = filename
                    documents.extend(docs)
                    print(f"Loaded text: {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
        
        return documents
    
    def create_vector_store(self, documents: List[Document]) -> Chroma:
        """Create and persist vector store from documents"""
        if not documents:
            raise ValueError("No documents to process")
        
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(chunks)} chunks")
        
        # Create vector store
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=config.CHROMA_PERSIST_DIR,
            collection_name=config.CHROMA_COLLECTION_NAME
        )
        
        print(f"Vector store created with {len(chunks)} chunks")
        return self.vector_store
    
    def load_existing_vector_store(self) -> Chroma:
        """Load existing vector store from disk"""
        if os.path.exists(config.CHROMA_PERSIST_DIR):
            self.vector_store = Chroma(
                persist_directory=config.CHROMA_PERSIST_DIR,
                embedding_function=self.embeddings,
                collection_name=config.CHROMA_COLLECTION_NAME
            )
            print("Loaded existing vector store")
            return self.vector_store
        else:
            raise FileNotFoundError("No existing vector store found")
    
    def search_documents(self, query: str, k: int = 5) -> List[Document]:
        """Search for relevant documents"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        
        return self.vector_store.similarity_search(query, k=k)