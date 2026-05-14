from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelResponse

from dotenv import load_dotenv

load_dotenv()

basic_model = ChatOpenAI(model = 'gpt-5.4-mini')
advanced_model = ChatOpenAI(model = 'gpt-5.4')

tools = []

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler):
    """Choose model based on conversation complexity ."""
    message_count = len(request.state['messages'])
    
    if message_count > 10:
        model = advanced_model
    else:
        model = basic_model
        
        return handler(request.override(model = model))
    
agent = create_agent(
    model = basic_model, 
    tools = tools,
    middleware = [dynamic_model_selection]
)
    
result = agent.invoke({
    'messages': [{'role': 'user', 'content': 'what is handler'}]
})
print(result['messages'][-1].content)