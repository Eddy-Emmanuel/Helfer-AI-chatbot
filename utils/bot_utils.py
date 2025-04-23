import os
from langchain_openai import ChatOpenAI
from config.secret_keys import project_config
from langchain_experimental.agents import create_pandas_dataframe_agent

os.environ["OPENAI_API_KEY"] = project_config.openai_api_key

llm = ChatOpenAI(model="gpt-4.1")

async def AnalyzeData(data):
    df_agent = create_pandas_dataframe_agent(llm=llm, 
                                             df=data,
                                             agent_type="tool-calling",
                                             verbose=False, 
                                             allow_dangerous_code=True)
    