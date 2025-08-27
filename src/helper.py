from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# extract data from PDF

def load_pdf(data):
    loader = DirectoryLoader(
        path=data, 
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    doc = loader.load()
    
    return doc


def text_split(extracted_data):
    spliter = RecursiveCharacterTextSplitter(
        chunk_size = 500, 
        chunk_overlap = 20
    )
    
    chunks = spliter.split_documents(extracted_data)
    
    return chunks

# embedding model
def load_embedding():
    embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embedding