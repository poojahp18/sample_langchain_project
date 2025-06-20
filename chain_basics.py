import os

from dotenv import load_dotenv
from langchain_openai.chat_models import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda
from langchain.schema.output_parser import StrOutputParser


load_dotenv(".env")


model = AzureChatOpenAI(
    azure_endpoint=os.environ.get("AZ_OPENAI_ENDPOINT"),
    api_key=os.environ.get("AZ_OPENAI_API_KEY"),
    azure_deployment=os.environ.get("AZ_OPENAI_DEPLOYMENT"),
    openai_api_type=os.environ.get("AZ_OPENAI_API_TYPE"),
    temperature=0,
)


# system : conetxt settinf of AI model human: direct prompts(queries) that need the response
animal_facts_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You like telling facts and you tell facts about {animal}"),
        ("human", "Tell me {count} facts."),
    ]
)

translation_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a translator and convert the provided text into {language}",
        ),
        ("human", "Translate the following text to {language}:{text}"),
    ]
)

prepare_for_translation = RunnableLambda(
    lambda output: {"text": output, "language": "English and Konkani"}
)

print(prepare_for_translation.invoke("Hello, how are you?"))

chain = (
    animal_facts_template
    | model
    | StrOutputParser()
    | prepare_for_translation
    | translation_template
    | model
    | StrOutputParser()
)

result = chain.invoke({"animal": "cat", "count": 2})

print(result)
