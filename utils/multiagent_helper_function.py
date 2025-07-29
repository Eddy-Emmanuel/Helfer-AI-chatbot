import os
from datetime import datetime
from typing import Type, List
from pydantic import BaseModel
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from config.secret_keys import project_config
from model.schema import AgentSchema, AgentRouter
from langgraph.graph import START, END, StateGraph
from langchain_core.runnables import RunnableLambda
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.agents import create_openai_tools_agent
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.agent_toolkits import SQLDatabaseToolkit

from config.secret_keys import project_config
os.environ["OPENAI_API_KEY"] = project_config.openai_api_key

def LoadDB(db_uri:Type[str], table_name:Type[List[str]]):
    db = SQLDatabase.from_uri(db_uri, include_tables=table_name)
    return db

def CreateAgent(db, llm, prompt):
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()
    agent = create_openai_tools_agent(llm, tools, prompt=prompt)
    return AgentExecutor(agent=agent, 
                         tools=tools, 
                         verbose=False,
                         handle_parsing_errors=True)

def CreateSearchAgent(llm):
    search = DuckDuckGoSearchResults()
    
    system_prompt = """
    You are a smart and efficient research assistant specialized in financial topics.
    Use the search tool when necessary to get updates from web.
    Always prioritize accuracy, clarity, and relevance in your answers.
    Think step-by-step when needed and explain your reasoning clearly. (Only when query explicitly says so.)
    Important: Always add the naira signs for currencys.
    """
    TEMPLATE = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("agent_scratchpad"),
        ("user", "{user_input}")
    ])
    
    search_tool = Tool(name="search_tool", 
                       func=lambda x:search.invoke(x), 
                       description="Search tool for getting real time information on the internet")
    
    agent = create_openai_tools_agent(llm=llm,
                                      tools=[search_tool],
                                      prompt=TEMPLATE)
    
    return AgentExecutor(agent=agent, tools=[search_tool], verbose=False)

class AgentTools:
    def __init__(self, llm, 
                 agent_schema, 
                 agent_router:Type[BaseModel],
                 db_uri:Type[str]):
        
        self.llm = llm
        self.agent_router = agent_router
        self.agent_schema = agent_schema
        self.helfer_db = LoadDB(db_uri=db_uri, table_name=["brands", "categories", "customers", "expense_accounts",
                                                           "expense_purposes", "expenses", "payment_methods", "point_of_sales", 
                                                           "products", "sales", "sales_items", "users", "shipping_methods", "units"])
        self.db_prompt = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

You MUST double check your query before executing it. If you get an error while
executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
database.

To start you should ALWAYS look at the tables in the database to see what you
can query. Do NOT skip this step.

Then you should query the schema of the most relevant tables.

Important: Always add the naira signs for currencies. 

Important: 
    - Avoiding exact string comparisons with '='; use flexible matching such as:
    - LOWER(TRIM(column)) LIKE LOWER(TRIM('%value%')) 
    - or ILIKE in systems that support it
    - Prefer `LIKE` or pattern matching for user-inputted strings
    - Use IN only if you're normalizing cases (e.g. LOWER(column) IN (...))

Important: The database contains multiple businesses. ALWAYS filter every SQL query using this condition: `business_id = '{business_id}'`.
"""

    def RouteQuery(self, state:AgentSchema):
        print("Routing query")
        llm_classifier = self.llm.with_structured_output(self.agent_router)
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an AI router. Based on the user's query, select the most appropriate agent.\n"
                "This is the information about each agents:\n"
                "'conversation_agent' for general dialogue,\n"
                "'helfer_db' for db related queries"
                "'search_agent' for price recommendations")
                    ),
            ("user", "{user_input}")
        ])
        chain = prompt|llm_classifier
        
        return chain.invoke({"user_input":state["user_query"]}).route

    def SearchAgent(self, state:AgentSchema):
        print("Invoking SearchAgent")
        agent = CreateSearchAgent(llm=self.llm)
        response = agent.invoke({"user_input":state["user_query"]})["output"]
        return {"agent_response" : response}

    def ConversationAgent(self, state:AgentSchema):
        print("Invoking ConversationAgent")
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a friendly helpful AI assistant. Provide clear, accurate, and helpful responses to user queries."),
            ("user", "{user_input}")
        ])
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=.9)
        chain = prompt|llm
        response = chain.invoke({"user_input":state["user_query"]})
        return {"agent_response":response.content}

    
    def HelferAgent(self, state: AgentSchema):
        print("Invoking HelferAgent")
        TEMPLATE = ChatPromptTemplate.from_messages([
            ("system", self.db_prompt.format(dialect=self.helfer_db.dialect, top_k=20, business_id=state["business_id"])),
            MessagesPlaceholder("agent_scratchpad"),
            ("user", "{user_input}")
        ])

        agent = CreateAgent(db=self.helfer_db, llm=self.llm, prompt=TEMPLATE)
    
        response = agent.invoke({"user_input":state["user_query"]})["output"]
    
        return {"agent_response" : response}

    def CompileAgent(self):
        graph = StateGraph(state_schema=self.agent_schema)

        graph.add_node("ConversationAgent", self.ConversationAgent)
        graph.add_node("SearchAgent", self.SearchAgent)
        graph.add_node("HelferAgent", self.HelferAgent)

        graph.add_conditional_edges(START,
                                    self.RouteQuery,
                                    {"conversation_agent":"ConversationAgent",
                                     "helfer_db":"HelferAgent",
                                     "search_agent":"SearchAgent"})

        graph.add_edge("ConversationAgent", END)
        graph.add_edge("HelferAgent", END)
        graph.add_edge("SearchAgent", END)
        
        return graph.compile()
    
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

helfercorps = AgentTools(llm=llm,
                         agent_schema=AgentSchema, 
                         agent_router=AgentRouter, 
                         db_uri=project_config.db_url)

helfercorps_bot = helfercorps.CompileAgent()

def add_context(state:AgentSchema):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    state["user_query"] = (
f"[Current Nigerian Time: {now}] [Model: helfercorps-v1] [Agent: Tobi]\n"
"This are your capabilities:\n"
"""
1. Generate Reports: Easily generate on-demand financial reports, stock summaries, and detailed sales breakdowns to stay informed and in control.\n
2. Monitor Inventory in Real-Time: Instantly check what’s in stock or out of stock by simply prompting Toby your AI buddy for quick inventory tracking.\n
3. Perform Sales Analysis: Toby helps you identify your best-selling products by time, day, or customer, and flags slow-moving or dead stock for smarter decision-making.\n
4. Price Suggestion: Get pricing recommendations based on competitor retail prices to stay competitive and maximize profit.\n
5. Business Health Insights: Track key performance indicators like profit margins, turnover rates, and overall business trends to make data-driven decisions.\n
6. Smart Data Entry & Upload: Automatically extract and organize data from PDFs or even handwritten notes directly into your system no manual input needed.
"""
"IMPORTANT: \nFormat your response properly. Highlight bullet points where necessary."
f"\nprevious_user_query:{state['previous_user_query']}\nprevious_ai_response:{state['previous_ai_response']}\nuser_query:{state['user_query']}"
)
    return state

helfercorps_bot_ed = RunnableLambda(add_context) | helfercorps_bot