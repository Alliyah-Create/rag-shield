import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings # <-- Back to OpenAI
from langchain_chroma import Chroma

# 1. Load our secret API key
load_dotenv()

def ingest_data():
    print("📂 Step 1: Loading document...")
    file_path = "data/company_policies.txt"
    
    if not os.path.exists(file_path):
        print(f"❌ ERROR: File not found at {file_path}")
        return
        
    loader = TextLoader(file_path)
    documents = loader.load()
    print(f"   -> Loaded {len(documents)} document(s).")

    print("✂️ Step 2: Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    chunks = text_splitter.split_documents(documents)
    print(f"   -> Created {len(chunks)} chunks of text.")
    
    if len(chunks) == 0:
        print("❌ ERROR: No chunks were created.")
        return

    print("🧠 Step 3: Generating embeddings (OpenAI) and saving to ChromaDB...")
    
    # Using OpenAI's embedding model
    embeddings = OpenAIEmbeddings()
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="company_policies",
        persist_directory="./chroma_db" 
    )
    print("✅ Success! Data securely ingested into Vector Database.")

if __name__ == "__main__":
    ingest_data()