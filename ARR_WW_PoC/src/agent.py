from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from .config import config

class DocumentSearchAgent:
    """Simplified agent for document search and question answering"""
    
    def __init__(self, vector_store_manager):
        self.vector_store_manager = vector_store_manager
        self.llm = ChatOpenAI(
            model=config.MODEL_NAME,
            temperature=config.TEMPERATURE,
            openai_api_key=config.CEREBRAS_API_KEY,
            openai_api_base=config.CEREBRAS_API_BASE,
            max_tokens=2000
        )
        self.conversation_history = []
    
    def _search_documents(self, query: str, k: int = 5) -> List[Dict]:
        """Search for relevant documents"""
        try:
            documents = self.vector_store_manager.search_documents(query, k)
            result = []
            for i, doc in enumerate(documents):
                result.append({
                    'content': doc.page_content,
                    'source': doc.metadata.get('source', 'Unknown'),
                    'page': doc.metadata.get('page', 'N/A'),
                    'index': i + 1
                })
            return result
        except Exception as e:
            return [{'error': f"Search failed: {str(e)}"}]
    
    def _summarize_context(self, documents: List[Dict]) -> str:
        """Summarize the retrieved documents"""
        try:
            if not documents or 'error' in documents[0]:
                return "No relevant documents found."
            
            # Combine document content
            context = "\n\n".join([doc['content'] for doc in documents])
            
            if len(context) > config.MAX_CONTEXT_LENGTH:
                context = context[:config.MAX_CONTEXT_LENGTH]
            
            system_prompt = """You are a helpful assistant that creates concise summaries. 
            Extract the key information from the provided text."""
            
            human_message = f"""Please summarize the following text:
            
            {context}
            
            Summary:"""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_message)
            ]
            
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def _generate_answer(self, question: str, context: str, documents: List[Dict]) -> str:
        """Generate final answer with citations"""
        try:
            if not documents or 'error' in documents[0]:
                return "I couldn't find enough relevant information in the documents to answer this question."
            
            # Prepare source information
            sources_info = []
            for doc in documents:
                if 'error' not in doc:
                    sources_info.append(f"[{doc['index']}] Source: {doc['source']}, Page: {doc['page']}")
            
            sources_text = "\n".join(sources_info)
            
            system_prompt = """You are a helpful assistant that provides accurate, grounded answers.
            Always base your answers strictly on the provided context.
            Cite your sources using the provided document numbers like [1], [2], etc."""
            
            human_message = f"""Question: {question}

            Context to use for answering:
            {context}

            Available sources for citation:
            {sources_text}

            Please provide a comprehensive answer based on the context above. 
            Include citations in your answer using the source numbers.

            Answer:"""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_message)
            ]
            
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a user query through the simplified workflow"""
        try:
            # Add to conversation history
            self.conversation_history.append(f"User: {query}")
            
            # Step 1: Search for relevant documents
            documents = self._search_documents(query)
            
            # Step 2: Summarize the context
            context = self._summarize_context(documents)
            
            # Step 3: Generate final answer
            answer = self._generate_answer(query, context, documents)
            
            # Add response to history
            self.conversation_history.append(f"Assistant: {answer}")
            
            return {
                "success": True,
                "answer": answer,
                "query": query,
                "documents_found": len([d for d in documents if 'error' not in d])
            }
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            self.conversation_history.append(f"Error: {error_msg}")
            return {
                "success": False,
                "answer": error_msg,
                "query": query,
                "documents_found": 0
            }
    
    def get_conversation_history(self) -> List[str]:
        """Get the conversation history"""
        return self.conversation_history.copy()