#!/usr/bin/env python3
"""
Simple runner script that handles imports correctly
"""

import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Main entry point"""
    print("🔍 Starting Agentic RAG Document Search System...")
    
    try:
        # Check if .env file exists
        if not os.path.exists('.env'):
            print("❌ Error: .env file not found.")
            print("💡 Please run: python setup.py")
            return
        
        # Import and run the main application
        from src.main import main as app_main
        app_main()
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()