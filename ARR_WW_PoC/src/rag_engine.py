from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate

from .config import config
from .vectorstore import VectorStoreManager

class RAGEngine:
    """Standard RAG engine for document search and question answering"""
    
    def __init__(self):
        self.vector_store_manager = VectorStoreManager()
        self.llm = ChatOpenAI(
            model=config.MODEL_NAME,
            temperature=config.TEMPERATURE,
            openai_api_key=config.CEREBRAS_API_KEY,
            openai_api_base=config.CEREBRAS_API_BASE,
            max_tokens=2000
        )
        
        # Initialize vector store
        self._initialize_vector_store()
        
        # Conversation history
        self.conversation_history = []
    
    def _initialize_vector_store(self):
        """Initialize or load the vector store"""
        try:
            if os.path.exists(config.CHROMA_PERSIST_DIR):
                self.vector_store_manager.load_existing_vector_store()
                print("✅ Loaded existing vector store")
            else:
                print("📁 Creating new vector store from documents...")
                documents = self.vector_store_manager.load_documents()
                if documents:
                    self.vector_store_manager.create_vector_store(documents)
                    print(f"✅ Vector store created with {len(documents)} documents")
                else:
                    print("⚠️  No documents found in data directory")
        except Exception as e:
            print(f"❌ Error initializing vector store: {e}")
    
    def _retrieve_documents(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve relevant documents for the query"""
        try:
            documents = self.vector_store_manager.search_documents(query, k)
            results = []
            for i, doc in enumerate(documents):
                results.append({
                    'content': doc.page_content,
                    'source': doc.metadata.get('source', 'Unknown'),
                    'page': doc.metadata.get('page', 'N/A'),
                    'index': i + 1
                })
            return results
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []
    
    def _generate_answer(self, query: str, documents: List[Dict]) -> Dict[str, Any]:
        """Generate answer based on retrieved documents"""
        try:
            if not documents:
                return {
                    "answer": "I couldn't find any relevant information in the documents to answer your question.",
                    "sources": []
                }
            
            # Prepare context from documents
            context_parts = []
            sources_info = []
            
            for doc in documents:
                context_parts.append(f"Source [{doc['index']}]: {doc['content']}")
                sources_info.append(f"[{doc['index']}] {doc['source']} (Page: {doc['page']})")
            
            context = "\n\n".join(context_parts)
            
            # Create prompt for answer generation
            system_prompt = """You are a helpful assistant that provides accurate answers based on the provided context.
            Always base your answers strictly on the provided documents.
            Cite your sources using the source numbers like [1], [2], etc. when referencing specific information.
            If the context doesn't contain enough information to answer the question, say so clearly.
            
            Important guidelines:
            - Be concise and factual
            - Only use information from the provided sources
            - Always cite your sources
            - If you're unsure, say you don't know based on the available information"""
            
            user_prompt = f"""Question: {query}

            Relevant Context:
            {context}

            Please provide a comprehensive answer based on the context above. Include citations for all factual information.

            Answer:"""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            return {
                "answer": response.content,
                "sources": sources_info
            }
            
        except Exception as e:
            return {
                "answer": f"Error generating answer: {str(e)}",
                "sources": []
            }
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a user query using standard RAG workflow"""
        try:
            # Add to conversation history
            self.conversation_history.append(f"User: {query}")
            
            # Step 1: Retrieve relevant documents
            documents = self._retrieve_documents(query)
            
            # Step 2: Generate answer based on retrieved documents
            result = self._generate_answer(query, documents)
            
            # Add response to history
            self.conversation_history.append(f"Assistant: {result['answer']}")
            
            return {
                "success": True,
                "answer": result['answer'],
                "sources": result['sources'],
                "documents_retrieved": len(documents),
                "query": query
            }
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            self.conversation_history.append(f"Error: {error_msg}")
            return {
                "success": False,
                "answer": error_msg,
                "sources": [],
                "documents_retrieved": 0,
                "query": query
            }
    
    def get_conversation_history(self) -> List[str]:
        """Get the conversation history"""
        return self.conversation_history.copy()

# For backward compatibility
DocumentSearchAgent = RAGEngine
