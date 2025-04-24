import os
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from config.secret_keys import project_config

os.environ["OPENAI_API_KEY"] = project_config.openai_api_key

llm = ChatOpenAI(model="gpt-4.1", temperature=0)
current_time = datetime.utcnow().strftime("%B %d, %Y, %H:%M UTC")

TEMPLATE =  f"""You are working with a pandas dataframe in Python. The name of the dataframe is df.
It is important to understand the attributes of the dataframe before working with it. This is the result of running df.head().to_markdown()

<df>
{{dhead}}
</df>

You are not meant to use only these rows to answer questions - they are meant as a way of telling you about the shape and schema of the dataframe.
You also do not have to use only the information here to answer questions - you can run intermediate queries to do exploratory data analysis to give you more information as needed.

The current date and time is {current_time}. Use this when interpreting dates or calculating durations.

For example:

<question>How old is Jane?</question>
<logic>Use person_name_search since you can use the query Jane</logic>
<question>Who has id 320</question>
<logic>Use python_repl since even though the question is about a person, you don't know their name so you can't include it.</logic>
"""

async def AnalyzeData(data, user_prompt):
    df_preview = data.head().to_markdown()
    template = TEMPLATE.format(dhead=df_preview)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", template),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
            ("human", "{input}"),
        ]
    )

    df_agent = create_pandas_dataframe_agent(llm=llm,
                                             df=data,
                                             agent_type="tool-calling",
                                             prompt=prompt,
                                             verbose=True,
                                             allow_dangerous_code=True)

    return df_agent.invoke(user_prompt)["output"]
