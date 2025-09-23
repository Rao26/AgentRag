import streamlit as st
import os
import sys

# Add the parent directory to the path so we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class StreamlitUI:
    """Streamlit interface for the Document Search Assistant"""
    
    def __init__(self):
        self.setup_page()
        self.vector_store_manager = None
        self.agent = None
        
    def setup_page(self):
        """Configure the Streamlit page"""
        st.set_page_config(
            page_title="Agentic RAG Document Search",
            page_icon="🔍",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🔍 Agentic RAG Document Search Assistant")
        st.markdown("""
        This intelligent assistant can search through your documents, break down complex queries, 
        and provide well-grounded answers with proper citations.
        """)
    
    def initialize_system(self):
        """Initialize the vector store and agent"""
        try:
            # Import inside the function to avoid circular imports
            from src.config import config
            from src.vectorstore import VectorStoreManager
            from src.agent import DocumentSearchAgent
            
            # Validate config first
            config.validate_config()
            
            with st.spinner("Initializing system..."):
                self.vector_store_manager = VectorStoreManager()
                
                # Try to load existing vector store, else create new one
                try:
                    self.vector_store_manager.load_existing_vector_store()
                    st.success("✅ Loaded existing vector store")
                except FileNotFoundError:
                    st.info("📁 No existing vector store found. Creating new one from documents...")
                    documents = self.vector_store_manager.load_documents()
                    if documents:
                        self.vector_store_manager.create_vector_store(documents)
                        st.success(f"✅ Created vector store with {len(documents)} documents")
                    else:
                        st.warning("⚠️ No documents found in data directory. Please add some PDF or text files.")
                        return False
                
                self.agent = DocumentSearchAgent(self.vector_store_manager)
                st.success("✅ Agent initialized successfully")
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
            
            if self.vector_store_manager and self.vector_store_manager.vector_store:
                try:
                    # Get document count from vector store
                    docs = self.vector_store_manager.vector_store.get()
                    doc_count = len(docs['documents']) if docs and 'documents' in docs else 0
                    st.metric("Chunks in Knowledge Base", doc_count)
                except Exception as e:
                    st.metric("Chunks in Knowledge Base", "Unknown")
            
            st.header("⚙️ Settings")
            st.info("Using Cerebras Llama-3.3-70b model")
            
            st.header("💡 Usage Tips")
            st.markdown("""
            - Ask specific questions for best results
            - The agent will automatically search, summarize, and answer
            - Answers include citations to source documents
            - Complex queries are broken down into subtasks
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
                {"role": "assistant", "content": "Hello! I'm your document search assistant. Ask me anything about the documents in your knowledge base."}
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
            
            # Generate agent response
            with st.chat_message("assistant"):
                with st.spinner("Searching documents and generating answer..."):
                    try:
                        result = self.agent.process_query(prompt)
                        
                        if result["success"]:
                            # Display answer
                            st.markdown(result["answer"])
                            
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