import os

from dotenv import load_dotenv
from langchain_openai.chat_models import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableParallel
from langchain.schema.output_parser import StrOutputParser


load_dotenv(".env")


model = AzureChatOpenAI(
    azure_endpoint=os.environ.get("AZ_OPENAI_ENDPOINT"),
    api_key=os.environ.get("AZ_OPENAI_API_KEY"),
    azure_deployment=os.environ.get("AZ_OPENAI_DEPLOYMENT"),
    openai_api_type=os.environ.get("AZ_OPENAI_API_TYPE"),
    temperature=0,
)

summary_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a movie critic"),
        ("human", "Provide brief summary of the movie {movie_name}."),
    ]
)


def analyze_plot(plot):
    plot_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a movie critic"),
            (
                "human",
                "analyze the plot {plot}. What are its 2 strengths and weaknesses.",
            ),
        ]
    )
    return plot_template.format_prompt(plot=plot)


def analyze_characters(characters):
    charater_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a movie critic"),
            (
                "human",
                "analyze the characters: {charaters}. What are its 1 strengths and weaknesses.",
            ),
        ]
    )
    return charater_template.format_prompt(charaters=characters)


def combine_verdicts(plot_analysis, character_analysis):
    return f"Plot Analysis:\n{plot_analysis}\n\nCharatcter Analysis:\n{character_analysis}\n"


plot_branch_chain = (
    RunnableLambda(lambda x: analyze_plot(x)) | model | StrOutputParser()
)

character_branch_chain = (
    RunnableLambda(lambda x: analyze_characters(x)) | model | StrOutputParser()
)

chain = (
    summary_template
    | model
    | StrOutputParser()
    | RunnableParallel(
        branches={"plot": plot_branch_chain, "characters": character_branch_chain}
    )
    | RunnableLambda(
        lambda x: combine_verdicts(x["branches"]["plot"], x["branches"]["characters"])
    )
)

result = chain.invoke({"movie_name": "Inception"})

print(result)
