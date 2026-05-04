from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from deepagents import create_deep_agent

load_dotenv()

pdf_paths = [
    "The Almanack of Naval.pdf",
    "Bhagavad gita.pdf",
    "Atomic Habits.pdf",
    "Durant Will - The Lessons of History.pdf",
    "How To Win Friends And Influence People - Carnegie, Dale.pdf",
    "Deep Work_ Rules for focused success in a distracted world - on Bookdio.org.pdf",
]

docs = []
for pdf_path in pdf_paths:
    loader = PyPDFLoader(pdf_path)
    loaded_docs = loader.load()
    
    for doc in loaded_docs:
        doc.metadata['source'] = pdf_path
        
    docs.extend(loaded_docs)

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

vector_stores = FAISS.from_documents(chunks, embeddings)

retriever = vector_stores.as_retriever(search_kwargs={'k':4})

@tool
def search_pdf(query: str):
    """Search the loaded PDF and return relevant text with page numbers."""
    found_docs = retriever.invoke(query)
    
    final = []
    
    for i, doc in enumerate(found_docs, start=1):
        page = doc.metadata.get("page")
        
        if page is not None:
            page = page + 1
        else:
            page = "unknown"
            
        final.append(
            f"\nSource {i} | Page {page}\n"
            f"{doc.page_content[:1000]}"
        )
    return "\n".join(final)

checkpointer = InMemorySaver()

agent = create_agent(
    model = 'openai:gpt-5.5',
    tools = [search_pdf],
    checkpointer = checkpointer,
    system_prompt = (
        'You are a PDF RAG assistant. '
        'Use the search_pdf tool when answering questions abou the pdf. '
        'Always include page numbers from the tool output. '
    ),
)

deep_agent = create_deep_agent(
    model = 'openai:gpt-4.1-mini',
    tools = [search_pdf],
    checkpointer = checkpointer,
    system_prompt = (
        'You are a deep research agent for PDF analysis. '
        'Plan when needed, search the PDF, synthesize clearly, '
        'and always include page-nummber evidence. '
    ),
)
# content = "What does the PDF say about influence people from book how to win friends and influenc people, keep it short in around 100-150 words ? Give answer with page citations."

# agent_result = agent.invoke(
#     {'messages': [{'role': 'user', 'content': content}]},
#     config={"configurable": {"thread_id": "my-pdf-chat-A"}},
# )

# deep_agent_result = deep_agent.invoke(
#     {'messages': [{'role': 'user', 'content': content}]},
#     config={"configurable": {"thread_id": "my-pdf-chat-2"}},
# )


# print("\n\n==== NORMAL AGENT RESULT ====\n")
# print(agent_result["messages"][-1].content)

# print("\n\n==== DEEP AGENT RESULT ====\n")
# print(deep_agent_result["messages"][-1].content)

thread_id = 'my-pdf-chat-A'

print("PDF saathi ready. Type 'exit' to stop. \n")

while True:
    user_input = input('You: ')
    
    if user_input.lower() in ['exit', 'quit', 'stop']:
        print('Chat ended')
        break
    
    agent_result = agent.invoke(
        {'messages': [{'role': 'user', 'content': user_input}]},
        config = {"configurable": {'thread_id': thread_id}},
    )
    
    print("\nAI:")
    print(agent_result["messages"][-1].content)
    print()