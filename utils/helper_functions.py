import os
from pydantic import BaseModel
from typing import Type, TypedDict
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from config.secret_keys import project_config
from config.schema import AgentSchema, AgentRouter
from langgraph.graph import START, END, StateGraph
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.agents import create_openai_tools_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit

from config.secret_keys import project_config
os.environ["OPENAI_API_KEY"] = project_config.openai_api_key

def LoadDB(db_uri, table_name):
    db = SQLDatabase.from_uri(db_uri, include_tables=[table_name])
    return db

def CreateAgent(db, llm, prompt):
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()
    agent = create_openai_tools_agent(llm, tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools)


class AgentTools:
    def __init__(self, llm, 
                 agent_schema:Type[TypedDict], 
                 agent_router:Type[BaseModel],
                 db_uri:Type[str]):
        
        self.llm = llm
        self.agent_router = agent_router
        self.agent_schema = agent_schema
        self.sales_db = LoadDB(db_uri=db_uri, table_name="sales")
        self.inventory_db = LoadDB(db_uri=db_uri, table_name="products")
        self.expense_db = LoadDB(db_uri=db_uri, table_name="expenses")
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

Important: The database contains multiple businesses. ALWAYS filter every SQL query using this condition: `business_id = '{business_id}'`.
"""

    def RouteQuery(self, state:AgentSchema):
        llm_classifier = self.llm.with_structured_output(self.agent_router)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI router. Based on the user's query, select the most appropriate agent."),
            ("user", "{user_input}")
        ])
        chain = prompt|llm_classifier
        
        return chain.invoke({"user_input":state["user_query"]}).route

    def ConversationAgent(self, state:AgentSchema):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Provide clear, accurate, and helpful responses to user queries."),
            ("user", "{user_input}")
        ])
        chain = prompt|self.llm
        response = chain.invoke({"user_input":state["user_query"]})
        return {"agent_response":response.content}

    def SalesAgent(self, state: AgentSchema):
        TEMPLATE = ChatPromptTemplate.from_messages([
            ("system", self.db_prompt.format(dialect=self.sales_db.dialect, top_k=5, business_id=state["business_id"])),
            MessagesPlaceholder("agent_scratchpad"),
            ("user", "{user_input}")
        ])

        agent = CreateAgent(db=self.sales_db, llm=self.llm, prompt=TEMPLATE)
    
        response = agent.invoke({"user_input":state["user_query"]})["output"]
    
        return {"agent_response" : response}

    def InventoryAgent(self, state:AgentSchema):
        TEMPLATE = ChatPromptTemplate.from_messages([
            ("system", self.db_prompt.format(dialect=self.inventory_db.dialect, top_k=5, business_id=state["business_id"])),
            MessagesPlaceholder("agent_scratchpad"),
            ("user", "{user_input}")
        ])

        agent = CreateAgent(db=self.inventory_db, llm=self.llm, prompt=TEMPLATE)
    
        response = agent.invoke({"user_input":state["user_query"]})["output"]
    
        return {"agent_response" : response}

    def ExpenseAgent(self, state:AgentSchema):
        TEMPLATE = ChatPromptTemplate.from_messages([
            ("system", self.db_prompt.format(dialect=self.expense_db.dialect, top_k=5, business_id=state["business_id"])),
            MessagesPlaceholder("agent_scratchpad"),
            ("user", "{user_input}")
        ])

        agent = CreateAgent(db=self.expense_db, llm=self.llm, prompt=TEMPLATE)
    
        response = agent.invoke({"user_input":state["user_query"]})["output"]
    
        return {"agent_response" : response}

    def CompileAgent(self):
        graph = StateGraph(state_schema=self.agent_schema)

        graph.add_node("ConversationAgent", self.ConversationAgent)
        graph.add_node("SalesAgent", self.SalesAgent)
        graph.add_node("InventoryAgent", self.InventoryAgent)
        graph.add_node("ExpenseAgent", self.ExpenseAgent)

        graph.add_conditional_edges(START,
                                    self.RouteQuery,
                                    {"conversation_agent":"ConversationAgent",
                                     "sales_agent":"SalesAgent",
                                     "inventory_agent":"InventoryAgent",
                                     "expense_agent":"ExpenseAgent"})

        graph.add_edge("ConversationAgent", END)
        graph.add_edge("SalesAgent", END)
        graph.add_edge("InventoryAgent", END)
        graph.add_edge("ExpenseAgent", END)
        
        return graph.compile()
    
llm = ChatOpenAI(model="gpt-4.1", temperature=0.9)

helfercorps = AgentTools(llm=llm, agent_schema=AgentSchema, 
                         agent_router=AgentRouter, 
                         db_uri=project_config.db_url)

helfercorps_bot = helfercorps.CompileAgent()