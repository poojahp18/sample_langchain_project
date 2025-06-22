import os

from dotenv import load_dotenv
from langchain_openai.chat_models import AzureChatOpenAI
from langchain.schema.messages import HumanMessage
from langchain.schema.runnable import RunnableSequence, RunnableLambda


load_dotenv(".env")


model = AzureChatOpenAI(
    azure_endpoint=os.environ.get("AZ_OPENAI_ENDPOINT"),
    api_key=os.environ.get("AZ_OPENAI_API_KEY"),
    azure_deployment=os.environ.get("AZ_OPENAI_DEPLOYMENT"),
    openai_api_type=os.environ.get("AZ_OPENAI_API_TYPE"),
    temperature=0,
)


# Create component functions
def clean_text(text):
    return text.strip().lower()


def format_message(text):
    return [HumanMessage(content=f"Summarize this: {text}")]


cleanText = RunnableLambda(clean_text)
formatMessage = RunnableLambda(format_message)

sequence = RunnableSequence(first=cleanText, middle=[formatMessage], last=model)

# Use the sequence
for chunk in sequence.stream(
    "  THIS is A TEST Message. THIS is A TEST Message. THIS is A TEST Message. THIS is A TEST Message. THIS is A TEST Message. THIS is A TEST Message.   "
):
    print(chunk.content, end="")
