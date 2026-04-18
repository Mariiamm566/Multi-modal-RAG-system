# Multi-Modal RAG QA System

A lightweight multi-modal Retrieval-Augmented Generation system for answering questions over PDF documents (text, tables, images).

## Features
- Text + Image retrieval (SentenceTransformers + CLIP)
- Cosine similarity ranking
- Source attribution (page-level)
- Streamlit demo

## Setup
pip install -r requirements.txt

## Run
streamlit run app.py

## Demo
1. Upload PDF
2. Click Build Index
3. Ask a question

## Example Queries
- cnn architecture diagram
- what is gradient descent

## Limitations
- No advanced reranking
- Limited table understanding
