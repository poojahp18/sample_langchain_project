from langchain.schema.runnable import RunnableLambda


def clean_text(text):
    return text.strip().lower()


def count_words(text):
    return len(text.split())


# Create a pipeline
pipeline = RunnableLambda(clean_text) | RunnableLambda(count_words)

result = pipeline.invoke("  Hello World  ")
print(result)

print(RunnableLambda(clean_text).invoke(" I'm here. Do you know me? "))
print(RunnableLambda(count_words).invoke(" I'm here. Do you know me? "))
