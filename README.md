# 🧠 AgadVed - Medical Chatbot
AgadVed is an AI-powered medical chatbot built to assist users with medical queries in a natural, conversational way. It combines modern NLP techniques with vector-based search and real-time speech recognition to deliver accurate, informative, and context-aware responses.

🌐 Live Demo: https://huggingface.co/spaces/jigyasa05/AgadVed_Medical_Chatbot

## 🚀 Features
💬 Chat with Medical Intelligence: Ask questions related to health, symptoms, diseases, or treatments.

🔍 RAG Pipeline: Combines retrieval from a custom medical dataset using Pinecone with generation using Groq LLM for precise answers.

🗣️ Voice Input: Use your microphone to ask questions via speech, transcribed in real-time using Whisper.

📄 Context-Aware Responses: Incorporates document-based retrieval with LangChain to ensure reliable answers.

🖥️ Gradio Interface: Clean and interactive frontend for easy chatting—via text or voice.

## 🛠️ Tech Stack
Frontend: Gradio

Backend: Python (FastAPI/Gradio)

Speech Recognition: OpenAI Whisper

Language Model: Groq API (LLM)

Retrieval: LangChain + Pinecone Vector Store

Embeddings: HuggingFace Sentence Transformers

Deployment: Hugging Face Spaces
