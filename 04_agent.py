from dotenv import load_dotenv

from langchain.agents import create_agent

load_dotenv()

def get_weather(city: str):
    """Return weather information for a given city."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model = "openai:gpt-4.1-mini",
    tools=[get_weather],
    system_prompt="You are a helpful ",
)

result = agent.invoke(
    {'messages':[{'role': 'user', 'content':"what's the weather in Berlin"}]}
)
print(result["messages"][-1].content_blocks)