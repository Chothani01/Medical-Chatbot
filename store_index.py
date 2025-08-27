from src.helper import load_pdf, text_split, load_embedding
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

load_dotenv()

doc = load_pdf("data")

chunks = text_split(doc)

embedding = load_embedding()

vectorstore = PineconeVectorStore.from_documents(documents=chunks, index_name='medicalchatbot', embedding=embedding)