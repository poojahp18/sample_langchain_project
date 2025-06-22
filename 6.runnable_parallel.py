import os

from dotenv import load_dotenv
from langchain_openai.chat_models import AzureChatOpenAI
from langchain.schema.runnable import RunnableParallel
from langchain.schema.messages import HumanMessage

load_dotenv(".env")


llm = AzureChatOpenAI(
    azure_endpoint=os.environ.get("AZ_OPENAI_ENDPOINT"),
    api_key=os.environ.get("AZ_OPENAI_API_KEY"),
    azure_deployment=os.environ.get("AZ_OPENAI_DEPLOYMENT"),
    openai_api_type=os.environ.get("AZ_OPENAI_API_TYPE"),
    temperature=0,
)


parallel_chain = RunnableParallel(
    summary=lambda x: llm.invoke(f"Summarize this: {x}"),
    translation=lambda x: llm.invoke(f"Translate to French: {x}"),
    sentiment=lambda x: llm.invoke(f"What's the sentiment of this text: {x}"),
)


# This will run all three LLM calls in parallel
result = parallel_chain.invoke("I love programming with Python!")

print("\nAll results:")
for key, value in result.items():
    print(f"{key}: {value.content}")

parallel_chain_with_content = RunnableParallel(
    summary=lambda x: llm.invoke(
        [HumanMessage(content=f"Summarize this: {x}")]
    ).content,
    translation=lambda x: llm.invoke(
        [HumanMessage(content=f"Translate to French: {x}")]
    ).content,
    sentiment=lambda x: llm.invoke(
        [HumanMessage(content=f"What's the sentiment of this text: {x}")]
    ).content,
)

result_with_content = parallel_chain_with_content.invoke(
    "I love programming with Python!"
)
print("\nDirect content results:", result_with_content)
