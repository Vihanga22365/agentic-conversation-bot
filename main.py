import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "Agentic Chatbot - RAG"

document_path = "./documents/"
loader = DirectoryLoader(document_path, show_progress=True)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)
db.save_local("faiss_db")
# db = FAISS.load_local("faiss_index", embeddings)
