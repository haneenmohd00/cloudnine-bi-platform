import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError(
        "GROQ_API_KEY was not found. Check your .env file."
    )

print("Groq API key found.")

llm = ChatGroq(
    # model="openai/gpt-oss-120b",
    model="groq/compound",
    temperature=0,
)

response = llm.invoke(
    "Reply with exactly: Groq connection successful."
)

print("\nGroq response:")
print(response.content)