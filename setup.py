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
        print("✅ Created data directory. Please add your PDF/text files here.")
    
    # Create sample data files
    sample_files = {
        'sample1.txt': """Artificial Intelligence Research Overview

Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions. The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving.

Key areas of AI research include machine learning, natural language processing, computer vision, and robotics. Machine learning algorithms use statistical techniques to give computers the ability to learn from data without being explicitly programmed.

Recent advancements in deep learning have significantly improved AI capabilities in areas like image recognition, speech recognition, and natural language understanding.""",
        
        'sample2.txt': """Machine Learning Implementation Guide

Machine Learning (ML) is a subset of artificial intelligence that focuses on building systems that learn from data. The three main types of machine learning are supervised learning, unsupervised learning, and reinforcement learning.

Supervised learning uses labeled datasets to train algorithms for classification or regression tasks. Common algorithms include linear regression, decision trees, and support vector machines.

Best practices for ML implementation include proper data preprocessing, feature engineering, model validation, and continuous monitoring. Data quality is critical for successful ML deployments."""
    }
    
    for filename, content in sample_files.items():
        filepath = os.path.join('data', filename)
        if not os.path.exists(filepath):
            print(f"📄 Creating sample file: {filename}")
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
    
    print("\n✅ Setup completed!")
    print("\n📋 Next steps:")
    print("1. Edit the .env file and add your CEREBRAS_API_KEY")
    print("2. Add your PDF/text files to the data/ directory")
    print("3. Run: streamlit run src/main.py")
    print("4. Or run CLI: python src/main.py --mode cli")

if __name__ == "__main__":
    setup_environment()