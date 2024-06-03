import streamlit as st
from typing import List
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from config import llm_open_router

#  Define the Joke and Jokes classes directly in the Streamlit app file
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

def generate_jokes(topic: str) -> List[Joke]:
    parser = PydanticOutputParser(pydantic_object=Jokes)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a top notch comedian. For any topic that the user suggests, give 10 jokes in the question punchline format.\nWrap the output in `json` tags\n{format_instructions}",
            ),
            ("human", "{query}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())
    
    chain = prompt | llm_open_router | parser

    response: Jokes = chain.invoke({"query": topic})
    
    return response.jokes

def main():
    st.title("Joke Generator")
    
    topic = st.text_input("Enter a topic for jokes:", "")
    generate_button = st.button("Generate Jokes")
    
    if generate_button and topic:
        jokes = generate_jokes(topic)
        
        if jokes:
            for i, joke in enumerate(jokes, start=1):
                with st.expander(f"Joke {i}: {joke.question}"):
                    st.write(joke.punchline)
        else:
            st.write("No jokes found for the given topic.")

if __name__ == "__main__":
    main()
    
    
# If you are facing the issue of adjustText not found, run the app with the full command
# `python3 -m streamlit run app/11_joke_generator_streamlit.py`
