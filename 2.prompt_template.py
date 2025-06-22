import os

from dotenv import load_dotenv
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv(".env")

llm = AzureChatOpenAI(
    azure_endpoint=os.environ.get("AZ_OPENAI_ENDPOINT"),
    api_key=os.environ.get("AZ_OPENAI_API_KEY"),
    azure_deployment=os.environ.get("AZ_OPENAI_DEPLOYMENT"),
    openai_api_type=os.environ.get("AZ_OPENAI_API_TYPE"),
    # api_version=os.environ.get("OPENAI_API_VERSION"),
    temperature=0,
)

# Promt for user understaning
template = "Write a {tone} email to {company} expressing interest in the {position}. Keep it to 4 points max"

# Convert abpve prompt to lang chain template
prompt_template = ChatPromptTemplate.from_template(template)

prompt = prompt_template.invoke(
    {"tone": "energetic", "company": "Bosch", "position": "AI Engineer"}
)
# print(prompt)

result = llm.invoke(prompt)

print(result.content)
