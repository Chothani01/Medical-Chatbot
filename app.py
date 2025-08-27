from flask import Flask, render_template, request
import os
from src.prompt import chain_type_kwargs
from src.helper import load_embedding
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import pinecone
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

llm = ChatGroq(model="openai/gpt-oss-120b")

embedding = load_embedding()

docSearch = PineconeVectorStore.from_existing_index(index_name="medicalchatbot", embedding=embedding)

llm = ChatGroq(model="openai/gpt-oss-120b")

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docSearch.as_retriever(search_kwargs={'k':2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result = qa.invoke({"query": input})
    print("Response: ", result["result"])
    return str(result["result"])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)