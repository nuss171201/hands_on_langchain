from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool

from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model = 'gpt-5.4',
    temperature = 0.1,
    max_tokens = 1000,
    timeout = 30)

agent = create_agent(model, tools = tools)

