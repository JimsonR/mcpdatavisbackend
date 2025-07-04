import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

if not all([api_key, endpoint, deployment, api_version]):
    raise ValueError("Missing one or more Azure OpenAI environment variables.")

client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=endpoint,
)

print(f"Testing Azure OpenAI LLM: {deployment} at {endpoint}")

try:
    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": "Say hello from Azure OpenAI!"}],
        max_tokens=50,
    )
    print("Response:", response.choices[0].message.content)
except Exception as e:
    print("Error communicating with Azure OpenAI:", e)
