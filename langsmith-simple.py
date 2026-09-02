import os
from dotenv import load_dotenv
from openai import OpenAI
from langsmith import wrappers, traceable

# Keys loaded from the project .env (never hardcode secrets).
load_dotenv()
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "")
LANGSMITH_KEY  = os.getenv("LANGSMITH_API_KEY", "")
if not OPENROUTER_KEY:
    raise SystemExit("Set OPENROUTER_API_KEY in your .env (copy .env.example -> .env).")

# Tracing is OPTIONAL: on only when a LangSmith key is present.
os.environ["LANGSMITH_TRACING"] = "true" if LANGSMITH_KEY else "false"
os.environ.setdefault("LANGSMITH_PROJECT", "langsmith-simple-demo")
if not LANGSMITH_KEY:
    print("[LangSmith key not set] The pipeline runs, but traces won't upload.\n"
          "Add LANGSMITH_API_KEY to your .env to watch the trace tree at "
          "https://smith.langchain.com\n")

client = wrappers.wrap_openai(OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_KEY,
))

@traceable
def ask_openai(topic: str):
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": f"Explain {topic} in simple terms."}]
    )
    return response.choices[0].message.content

@traceable
def ask_claude(openai_output: str):
    response = client.chat.completions.create(
        model="anthropic/claude-3-haiku",
        messages=[{"role": "user", "content": f"Summarize this in one line: {openai_output}"}]
    )
    return response.choices[0].message.content

@traceable
def pipeline(topic: str):
    step1 = ask_openai(topic)
    print(f"OpenAI explanation: {step1}\n")

    step2 = ask_claude(step1)
    print(f"Claude summary: {step2}")

pipeline("black holes")