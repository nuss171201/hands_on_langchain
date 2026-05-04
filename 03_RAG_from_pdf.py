
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA

load_dotenv()

pdf_path = "The Almanack of Naval.pdf"
pdf_path = "Bhagavad gita.pdf"

loader = PyPDFLoader(pdf_path)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(model = 'text-embedding-3-small')

vector_store = FAISS.from_documents(chunks, embeddings)

retriever = vector_store.as_retriever(search_kwargs = {'k':4})

llm = ChatOpenAI(model = 'gpt-5.5')

qa = RetrievalQA.from_chain_type(
    llm = llm,
    retriever = retriever
)

result = qa.invoke({
    'query': 'What this pdf talks about medetation ?'
})

print(result['result'])