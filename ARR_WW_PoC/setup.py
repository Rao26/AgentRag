#!/usr/bin/env python3
"""
Setup script for Agentic RAG System
"""

import os
import sys

def setup_environment():
    """Setup the environment for the RAG system"""
    print("🔧 Setting up Agentic RAG System...")
    print("=" * 50)
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("📝 Creating .env file...")
        with open('.env', 'w') as f:
            f.write('# Cerebras API Configuration\n')
            f.write('CEREBRAS_API_KEY=your_api_key_here\n\n')
            f.write('# Get your API key from: https://www.cerebras.net/\n')
        print("✅ Created .env file. Please edit it with your actual API key.")
    
    # Check if data directory exists
    if not os.path.exists('data'):
        print("📁 Creating data directory...")
        os.makedirs('data')
        print("✅ Created data directory.")
    
    # Create instruction file for the Bhutan climate change PDF
    instruction_content = """RAG System Setup Instructions

This RAG system is configured to work with the Bhutan climate change document:

"2011-NEC-climate_change-pub.pdf" - National Environment Commission, Royal Government of Bhutan

To use this system:

1. Place the PDF file '2011-NEC-climate_change-pub.pdf' in the data/ directory
2. The document contains comprehensive information about:
   - Climate change science and evidence
   - Greenhouse gases and their effects
   - Impacts on Bhutan and Himalayan region
   - Glacial retreat and water resources
   - Adaptation and mitigation strategies
   - Bhutan's climate policies and carbon neutrality commitment

3. Once the PDF is placed in the data/ directory, run the system using:
   - Streamlit: streamlit run src/main.py
   - CLI: python src/main.py --mode cli

The system will process this document and allow you to ask questions about climate change, Bhutan's environment, glacial lakes, and related topics.
"""

    instruction_file = os.path.join('data', 'INSTRUCTIONS.md')
    if not os.path.exists(instruction_file):
        print("📄 Creating instruction file...")
        with open(instruction_file, 'w', encoding='utf-8') as f:
            f.write(instruction_content)
    
    print("\n✅ Setup completed!")
    print("\n📋 Next steps:")
    print("1. Edit the .env file and add your CEREBRAS_API_KEY")
    print("2. Add '2011-NEC-climate_change-pub.pdf' to the data/ directory")
    print("3. Run: streamlit run src/main.py")
    print("4. Or run CLI: python src/main.py --mode cli")
    print("\n📚 This system is specifically configured for the Bhutan climate change document.")

if __name__ == "__main__":
    setup_environment()
