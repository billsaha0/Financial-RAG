import os
import qdrant_client
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

def create_vector_database():
    print("Initializing AI Database Setup with Local Embeddings...")
    
    print("Loading HuggingFace BGE-Small Embedding Model...")
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    file_path = "parsed_output.md"
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found")
        return
        
    print("Loading parsed document...")
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()

    parser = MarkdownNodeParser()
    nodes = parser.get_nodes_from_documents(documents)
    print(f"Document split into {len(nodes)} logical chunks.")

    client = qdrant_client.QdrantClient(path="./qdrant_db")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="financial_reports",
        enable_hybrid=True
        )
    
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("Generating embeddings locally and saving to Qdrant database...")
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        show_progress=True
    )
    
    print("Success! Your Vector Database is built and ready.")

if __name__ == "__main__":
    create_vector_database()