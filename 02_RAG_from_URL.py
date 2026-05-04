from dotenv import load_dotenv


from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter                         # splitter = machine/tool prepared to cut long documents into smaller overlapping parts
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS                                          # FAISS stores embeddings/vectors and searches the closest ones
# from langchain.chains import RetrievalQA
from langchain_classic.chains import RetrievalQA


load_dotenv()

# url = 'https://www.gutenberg.org/files/64317/64317-0.txt'
url = 'https://www.levels.fyi/?tab=levels'


loader = WebBaseLoader(url)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(model ='text-embedding-3-small')

vector_store = FAISS.from_documents(chunks, embeddings)

retriever = vector_store.as_retriever(search_kwargs={'k':3})

llm = ChatOpenAI(model='gpt-4.1-mini')

qa = RetrievalQA.from_chain_type(
    llm = llm,
    retriever = retriever
)

result = qa.invoke({
    'query': "Which jobs pays the most, and how much they pays"
})

print(result['result'])