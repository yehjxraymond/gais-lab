from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd
from config import llm_open_router

df = pd.read_csv(
    "https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv"
)

def main():
    agent = create_pandas_dataframe_agent(
        llm_open_router,
        df,
        verbose=True,
        allow_dangerous_code=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
    )
    agent.invoke("how many rows are there?")
    # agent.invoke("what are the columns of the dataframe?")
    # agent.invoke("what are the columns of the dataframe? also can you give me the first 5 rows?")

if __name__ == "__main__":
    main()
