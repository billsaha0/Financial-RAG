import os
import streamlit as st
import qdrant_client
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core.postprocessor import SentenceTransformerRerank

load_dotenv()

st.set_page_config(page_title="Financial RAG")
st.title("Financial-RAG Assistant")
st.markdown("Ask questions about your financial reports.")

@st.cache_resource
def initialize_ai():
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    if not os.environ.get("GROQ_API_KEY"):
        raise ValueError("Missing GROQ_API_KEY in environment variables")
    Settings.llm = Groq(model="llama-3.3-70b-versatile", api_key=os.environ.get("GROQ_API_KEY"))
    
    client = qdrant_client.QdrantClient(path="./qdrant_db")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="financial_reports",
        enable_hybrid=True
        )
    
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    reranker = SentenceTransformerRerank(
        model="BAAI/bge-reranker-base", 
        top_n=3
    )

    return index.as_chat_engine(
        chat_mode="condense_question",
        verbose=True,
        vector_store_query_mode="hybrid",
        similarity_top_k=10,
        sparse_top_k=10,
        node_postprocessors=[reranker]
        )

try:
    chat_engine = initialize_ai()
except Exception as e:
    st.error(f"Failed to load the AI: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I have analyzed your financial documents. What would you like to know?"}
    ]

with st.sidebar:
    st.header("Settings")
    if st.button("Clear Conversation"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I have analyzed your financial documents. What would you like to know?"}
        ]
        st.rerun()
    
    st.divider()
    st.markdown("**System Status:**")
    st.markdown("Qdrant DB Connected")
    st.markdown("Llama 3.3 Active")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask anything about your financial documents"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        streaming_response = chat_engine.stream_chat(prompt)
        
        for token in streaming_response.response_gen:
            full_response += token
            message_placeholder.markdown(full_response + "▌")
            
        message_placeholder.markdown(full_response)
        
        source_nodes = streaming_response.source_nodes
        if source_nodes:
            with st.expander("View Source Documents"):
                for i, node in enumerate(source_nodes):
                    st.markdown(f"**Source {i+1}:**")
                    st.info(node.text)
                    
        st.session_state.messages.append({"role": "assistant", "content": full_response})