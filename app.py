import streamlit as st
from langchain_openai import OpenAI
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from chromadb.config import Settings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Streamlit app title
st.title("RAG Chatbot of Any Website by Anas Akhtar")

# URLs to process
urls = [
    'https://en.wikipedia.org/wiki/Large_language_model',
    'https://www.victoriaonmove.com.au/local-removalists.html',
    'https://victoriaonmove.com.au/index.html',
    'https://victoriaonmove.com.au/contact.html',
    'https://techwithwarriors.com/'
]

# Load documents
loader = UnstructuredURLLoader(urls=urls)
data = loader.load()

# Split documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)
all_splits = docs

# Initialize Chroma with the new configuration
persist_directory = "./chroma_db"  # Directory for storing the Chroma database
vectorstore = Chroma.from_documents(
    documents=all_splits,
    embedding=OpenAIEmbeddings(),
    persist_directory=persist_directory
)

# Set up the retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# Initialize the language model
llm = OpenAI(temperature=0.4, max_tokens=500)

# Create a query input box
query = st.chat_input("Ask me anything: ")

# System prompt for the chatbot
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

# Chat prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Process the query
if query:
    # Create the retrieval-augmented generation chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Get the response
    response = rag_chain.invoke({"input": query})

    # Display the response
    st.write(response["answer"])
