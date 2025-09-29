#!/usr/bin/env python3
"""
Main entry point for the RAG Document Search System
"""

import argparse
import sys
import os

# Add the parent directory to the path so we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="RAG Document Search System")
    parser.add_argument(
        '--mode', 
        choices=['cli', 'web'], 
        default='web',
        help='Run in CLI mode or web UI mode (default: web)'
    )
    
    args = parser.parse_args()
    
    try:
        # Validate configuration first
        from src.config import config
        config.validate_config()
        
        if args.mode == 'cli':
            command_line_interface()
        else:
            # Import and run Streamlit UI
            from src.ui import StreamlitUI
            ui = StreamlitUI()
            ui.run()
            
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        print("\n💡 Solution: Create a .env file in the project root with your CEREBRAS_API_KEY")
        sys.exit(1)
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Solution: Make sure all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        sys.exit(1)

def command_line_interface():
    """Run the system in command line mode"""
    print("🔍 RAG Document Search - CLI Mode")
    print("=" * 50)
    
    try:
        from src.rag_engine import RAGEngine
        
        rag_engine = RAGEngine()
        
        print("\n✅ System ready! Type your questions below (or 'quit' to exit):")
        print("-" * 50)
        
        while True:
            query = input("\n🤔 Question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye! 👋")
                break
            
            if not query:
                continue
            
            print("🔄 Processing...")
            result = rag_engine.process_query(query)
            
            if result["success"]:
                print(f"\n🤖 Answer: {result['answer']}")
                if result['sources']:
                    print(f"\n📚 Sources:")
                    for source in result['sources']:
                        print(f"  • {source}")
                print("-" * 50)
            else:
                print(f"\n❌ Error: {result['answer']}")
                
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
