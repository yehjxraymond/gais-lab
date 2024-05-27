from typing import List

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from config import llm_open_router

class Joke(BaseModel):
    """A joke in the question punchline format."""
    
    question: str = Field(
        description="The question part of the joke.",
        example="Why did the generative AI break up with its girlfriend?"
    )
    punchline: str = Field(
        description="The punchline part of the joke.",
        example="It couldn't handle the long distance relationships."
    )
    
class Jokes(BaseModel):
    """List of jokes."""
    
    jokes: List[Joke]
    

def main():
    parser = PydanticOutputParser(pydantic_object=Jokes)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a top notch comedian. For any topic that the user suggest give 10 jokes in the question punchline format.\nWrap the output in `json` tags\n{format_instructions}",
            ),
            ("human", "{query}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())
    query = "Generative AI"
    
    print("Prompt to be sent to the LLM:", prompt.invoke({"query": query}))

    chain = prompt | llm_open_router | parser

    response: Jokes = chain.invoke({"query": query})    
    
    # Iterate through each joke in the response and print it
    for joke in response.jokes:
        print(f"Q: {joke.question}")
        print(f"A: {joke.punchline}\n")


if __name__ == "__main__":
    main()

# https://python.langchain.com/v0.2/docs/how_to/structured_output/#using-pydanticoutputparser