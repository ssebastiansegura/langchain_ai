import os

from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

## TOOL Gemini will use whenever it needs it.
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"I'ts always sunny in {city}!"

GENAI_MODEL = "gemini-2.5-flash-lite"
GENAI_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model=GENAI_MODEL, 
    google_api_key=GENAI_API_KEY
)

agent = create_agent(
    model=llm,                                      ## agent's model (gemini-2.5-flash-lite).
    tools=[get_weather],                            ## tools it is allowed to use.
    system_prompt="You are a helpful assistant.",   ## personality.
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
)
print(result["messages"][-1].content)