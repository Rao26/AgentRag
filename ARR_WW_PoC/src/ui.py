import streamlit as st
import os
import sys

# Add the parent directory to the path so we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class StreamlitUI:
    """Streamlit interface for the RAG Document Search Assistant"""
    
    def __init__(self):
        self.setup_page()
        self.rag_engine = None
        
    def setup_page(self):
        """Configure the Streamlit page"""
        st.set_page_config(
            page_title="RAG Document Search",
            page_icon="🔍",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🔍 RAG Document Search Assistant")
        st.markdown("""
        This intelligent assistant can search through your documents and provide 
        well-grounded answers with proper citations using Retrieval-Augmented Generation.
        """)
    
    def initialize_system(self):
        """Initialize the RAG engine"""
        try:
            from src.config import config
            from src.rag_engine import RAGEngine
            
            # Validate config first
            config.validate_config()
            
            with st.spinner("Initializing RAG system..."):
                self.rag_engine = RAGEngine()
                st.success("✅ RAG system initialized successfully")
                return True
                
        except ValueError as e:
            st.error(f"❌ Configuration Error: {e}")
            st.info("💡 Please create a `.env` file in the project root with your `CEREBRAS_API_KEY`")
            return False
        except Exception as e:
            st.error(f"❌ Error initializing system: {str(e)}")
            return False
    
    def display_sidebar(self):
        """Display the sidebar with information and controls"""
        with st.sidebar:
            st.header("📊 System Information")
            
            if self.rag_engine and self.rag_engine.vector_store_manager.vector_store:
                try:
                    docs = self.rag_engine.vector_store_manager.vector_store.get()
                    doc_count = len(docs['documents']) if docs and 'documents' in docs else 0
                    st.metric("Document Chunks", doc_count)
                except Exception as e:
                    st.metric("Document Chunks", "Unknown")
            
            st.header("⚙️ Settings")
            st.info("Using RAG with Cerebras Llama-3.3-70b")
            
            st.header("💡 Usage Tips")
            st.markdown("""
            - Ask specific questions for best results
            - Answers include citations to source documents
            - System retrieves relevant documents and generates answers
            - Supports PDF and text documents
            """)
            
            # Clear conversation button
            if st.button("🔄 Clear Conversation"):
                if "messages" in st.session_state:
                    st.session_state.messages = []
                st.rerun()
    
    def display_chat_interface(self):
        """Display the main chat interface"""
        st.header("💬 Chat with Your Documents")
        
        # Initialize chat history in session state
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello! I'm your RAG document search assistant. Ask me anything about the documents in your knowledge base."}
            ]
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate RAG response
            with st.chat_message("assistant"):
                with st.spinner("Searching documents and generating answer..."):
                    try:
                        result = self.rag_engine.process_query(prompt)
                        
                        if result["success"]:
                            # Display answer
                            st.markdown(result["answer"])
                            
                            # Display sources if available
                            if result["sources"]:
                                with st.expander("📚 Source References"):
                                    for source in result["sources"]:
                                        st.write(f"• {source}")
                            
                            # Add assistant response to chat history
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": result["answer"]
                            })
                        else:
                            st.error(result["answer"])
                    except Exception as e:
                        error_msg = f"Error processing query: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": error_msg
                        })
    
    def run(self):
        """Run the Streamlit application"""
        if self.initialize_system():
            self.display_sidebar()
            self.display_chat_interface()
        else:
            st.error("Failed to initialize the system. Please check the errors above.")

def main():
    ui = StreamlitUI()
    ui.run()

if __name__ == "__main__":
    main()
