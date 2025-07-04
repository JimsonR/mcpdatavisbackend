import os
from dotenv import load_dotenv
import openai

load_dotenv(".env")

api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")

print("Testing Azure OpenAI credentials...")
print("DEPLOYMENT:", deployment)
print("ENDPOINT:", endpoint)
print("API_KEY:", api_key)
print("API_VERSION:", api_version)

client = openai.AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=endpoint,
)

try:
    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=10
    )
    print("SUCCESS! Response:")
    print(response)
except Exception as e:
    print("FAILED:", e)