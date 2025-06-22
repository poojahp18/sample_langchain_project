import os

from langchain_openai.chat_models import AzureChatOpenAI
from dotenv import load_dotenv

load_dotenv(".env")

llm = AzureChatOpenAI(
    azure_endpoint=os.environ.get("AZ_OPENAI_ENDPOINT"),
    api_key=os.environ.get("AZ_OPENAI_API_KEY"),
    azure_deployment=os.environ.get("AZ_OPENAI_DEPLOYMENT"),
    openai_api_type=os.environ.get("AZ_OPENAI_API_TYPE"),
    # api_version=os.environ.get("OPENAI_API_VERSION"),
    temperature=0,
)

result = llm.invoke(
    "What is the best way to learn something new. Provide it in 5 points"
)

print(result.content)
import os

from langchain_openai.chat_models import AzureChatOpenAI
from dotenv import load_dotenv

load_dotenv(".env")

llm = AzureChatOpenAI(
    azure_endpoint=os.environ.get("AZ_OPENAI_ENDPOINT"),
    api_key=os.environ.get("AZ_OPENAI_API_KEY"),
    azure_deployment=os.environ.get("AZ_OPENAI_DEPLOYMENT"),
    openai_api_type=os.environ.get("AZ_OPENAI_API_TYPE"),
    # api_version=os.environ.get("OPENAI_API_VERSION"),
    temperature=0,
)

result = llm.invoke(
    "What is the best way to learn something new. Provide it in 5 points"
)

print(result.content)
