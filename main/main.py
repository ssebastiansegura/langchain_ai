import os
import urllib.request ## handles making HTTP requests.
import urllib.error   ## handles errors from HTTP requests.

from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import create_agent  

GENAI_MODEL = "gemini-2.5-flash-lite"
GENAI_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model=GENAI_MODEL, 
    google_api_key=GENAI_API_KEY
)

## TOOL Gemini will use whenever it needs it.
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"I'ts always sunny in {city}!"

@tool ## registers function as tool
def fetch_text_from_url(url: str) -> str:
    """Fetch the document from a URL."""
    ## create HTTP request, Pyhtin script dressed up as a browser (Mozilla/5.0) to access website.
    req = urllib.request.Request (
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; quickstart-research/1.0)"},
    )
    ## attempts to open URL and read its raw content.
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
    ## error handling
    except urllib.error.URLError as e:
        return f"Fetch failed: {e}"
    ## converts into readable Python string, if any ch can't be decoded, it replaces it with "?" instead of crashing
    text = raw.decode("utf-8", errors="replace")
    return text


SYSTEM_PROMPT = """You are a literary data assistant.

## Capabilities

- `fetch_text_from_url`: loads document text from a URL into the conversation.
Do not guess line counts or positions—ground them in tool results from the saved file."""

agent = create_agent(
    model=llm,                                      ## agent's model (gemini-2.5-flash-lite).
    tools=[get_weather, fetch_text_from_url],       ## tools it is allowed to use.
    system_prompt=SYSTEM_PROMPT,                    ## personality.
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Give me a summary of the information you found on this website please: https://www.accuweather.com/en/cr/heredia/113006/weather-forecast/113006"}]}
)
print(result["messages"][-1].content)