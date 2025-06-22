from langchain.schema.runnable import RunnableLambda


# Create a simple function
def multiply_by_two(x):
    return x * 2


# Wrap it with RunnableLambda
runnable = RunnableLambda(multiply_by_two)

# Use it
result = runnable.invoke(5)  # Returns 10
print(result)
