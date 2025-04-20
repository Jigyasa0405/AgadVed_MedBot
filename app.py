import gradio as gr
from dotenv import load_dotenv
import os
from datetime import datetime
from src.helper import download_hugging_face_embeddings, transcribe_audio
from src.prompt import system_prompt
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# API setup
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Embedding and retriever setup
embeddings = download_hugging_face_embeddings()
docsearch = PineconeVectorStore.from_existing_index(index_name="medicalbot", embedding=embeddings)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# LLM and chains
llm = ChatGroq(api_key=GROQ_API_KEY, model_name="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.4)
prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)


def get_chatbot_response(msg):
    try:
        response = rag_chain.invoke({"input": msg})
        return str(response["answer"])
    except Exception as e:
        return "Sorry, something went wrong while fetching the response. Please try again later."


# Main chat function
def chat(text_input, state):
    response = get_chatbot_response(text_input)
    state.append({"role": "user", "content": text_input})
    state.append({"role": "assistant", "content": response})
    return state, state


# Transcribe + Chat
def handle_audio(audio_input, state):
    text_input = transcribe_audio(audio_input)
    response = get_chatbot_response(text_input)
    state.append({"role": "user", "content": text_input})
    state.append({"role": "assistant", "content": response})
    return state, state


# Custom CSS for sky blue background
custom_css = """
.gradio-container {
    background-color: #87CEEB !important;
}
"""

# Interface with voice and text
with gr.Blocks(title="AgadVed - Medical Chatbot", css=custom_css) as demo:
    gr.Markdown("## 🩺 AgadVed - Medical Chatbot")
    chatbot = gr.Chatbot(type="messages", label="AgadVed", bubble_full_width=False)
    state = gr.State([])

    with gr.Row():
        with gr.Column(scale=5):
            text_input = gr.Textbox(placeholder="Type your medical question...")
        with gr.Column(scale=1):
            send_btn = gr.Button("Send")

    with gr.Row():
        audio_input = gr.Audio(type="filepath", label="🎙️ Speak your question")
        voice_btn = gr.Button("Transcribe & Send")

    send_btn.click(chat, inputs=[text_input, state], outputs=[chatbot, state])
    voice_btn.click(handle_audio, inputs=[audio_input, state], outputs=[chatbot, state])

demo.launch()
