import os
from datetime import datetime
from typing import Type, List
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain.agents import AgentExecutor
from config.secret_keys import project_config
from model.schema import AgentSchema, AgentRouter
from langgraph.graph import START, END, StateGraph
from langchain_core.runnables import RunnableLambda
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.agents import create_openai_tools_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit

from config.secret_keys import project_config
os.environ["OPENAI_API_KEY"] = project_config.openai_api_key
os.environ["TAVILY_API_KEY"] = project_config.tavily_api_key

def LoadDB(db_uri:Type[str], table_name:Type[List[str]]):
    db = SQLDatabase.from_uri(db_uri, include_tables=table_name)
    return db

def CreateAgent(db, llm, prompt):
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()
    agent = create_openai_tools_agent(llm, tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools)

def CreateSearchAgent(llm):
    search_tool = TavilySearch(max_results=5,
                               topic="general",
                               search_depth="basic",
                               time_range="month")
    system_prompt = """
    You are a smart and efficient research assistant specialized in financial topics.
    Use the search tool when necessary to gather the latest insights, news, or updates from the past month.
    Always prioritize accuracy, clarity, and relevance in your answers.
    Think step-by-step when needed and explain your reasoning clearly. (Only when query explicitly says so.)
    Important: Always add the naira signs for currencys.
    """
    TEMPLATE = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("agent_scratchpad"),
        ("user", "{user_input}")
    ])
    agent = create_openai_tools_agent(llm=llm, tools=[search_tool], prompt=TEMPLATE)
    return AgentExecutor(agent=agent, tools=[search_tool], verbose=False)

class AgentTools:
    def __init__(self, llm, 
                 agent_schema, 
                 agent_router:Type[BaseModel],
                 db_uri:Type[str]):
        
        self.llm = llm
        self.agent_router = agent_router
        self.agent_schema = agent_schema
        self.customer_sales_db = LoadDB(db_uri=db_uri, table_name=["sales", "sales_items"])
        self.inventory_db = LoadDB(db_uri=db_uri, table_name=["products"])
        self.expense_db = LoadDB(db_uri=db_uri, table_name=['expense_accounts', 'expense_purposes', 'expenses'])
        self.customer_db = LoadDB(db_uri=db_uri, table_name=["customers"])
        self.pointOfSales_db = LoadDB(db_uri=db_uri, table_name=["point_of_sales"])
        self.paymentmethod_db = LoadDB(db_uri=db_uri, table_name=["payment_methods"])
        self.categories_and_brand_db = LoadDB(db_uri=db_uri, table_name=['brands', 'categories'])
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

Important: Always add the naira signs for currencys.

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
                "'customer_sales_agent' for questions related to customer sales,\n"
                "'inventory_agent' for inventory analysis,\n"
                "'expense_agent' for expense tracking and management,\n"
                "'search_agent' for price recommendations,\n"
                "'point_of_sales_agent' for queries related to point of sales and walk-in sales,\n"
                "'payment_methods_agent' for queries related to payment methods,\n"
                "'category&brand_agent' for queries related to selling categories and selling brands.")
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
        llm = ChatOpenAI(model="gpt-4.1", temperature=.9)
        chain = prompt|llm
        response = chain.invoke({"user_input":state["user_query"]})
        return {"agent_response":response.content}

    def CustomerSalesAgent(self, state: AgentSchema):
        print("Invoking SalesAgent")
        TEMPLATE = ChatPromptTemplate.from_messages([
            ("system", self.db_prompt.format(dialect=self.customer_sales_db.dialect, top_k=20, business_id=state["business_id"])),
            MessagesPlaceholder("agent_scratchpad"),
            ("user", "{user_input}")
        ])

        agent = CreateAgent(db=self.customer_sales_db, llm=self.llm, prompt=TEMPLATE)
    
        response = agent.invoke({"user_input":state["user_query"]})["output"]
    
        return {"agent_response" : response}

    def InventoryAgent(self, state:AgentSchema):
        print("Invoking InventoryAgent")
        TEMPLATE = ChatPromptTemplate.from_messages([
            ("system", self.db_prompt.format(dialect=self.inventory_db.dialect, top_k=20, business_id=state["business_id"])),
            MessagesPlaceholder("agent_scratchpad"),
            ("user", "{user_input}")
        ])

        agent = CreateAgent(db=self.inventory_db, llm=self.llm, prompt=TEMPLATE)
    
        response = agent.invoke({"user_input":state["user_query"]})["output"]
    
        return {"agent_response" : response}

    def ExpenseAgent(self, state:AgentSchema):
        print("Invoking ExpenseAgent")
        TEMPLATE = ChatPromptTemplate.from_messages([
            ("system", self.db_prompt.format(dialect=self.expense_db.dialect, top_k=20, business_id=state["business_id"])),
            MessagesPlaceholder("agent_scratchpad"),
            ("user", "{user_input}")
        ])

        agent = CreateAgent(db=self.expense_db, llm=self.llm, prompt=TEMPLATE)
    
        response = agent.invoke({"user_input":state["user_query"]})["output"]
    
        return {"agent_response" : response}
    
    def PointOfSalesAgent(self, state:AgentSchema):
        print("Invoking PointOfSalesAgent")
        TEMPLATE = ChatPromptTemplate.from_messages([
            ("system", self.db_prompt.format(dialect=self.pointOfSales_db.dialect, top_k=20, business_id=state["business_id"])),
            MessagesPlaceholder("agent_scratchpad"),
            ("user", "{user_input}")
        ])

        agent = CreateAgent(db=self.pointOfSales_db, llm=self.llm, prompt=TEMPLATE)
    
        response = agent.invoke({"user_input":state["user_query"]})["output"]
    
        return {"agent_response" : response}
    
    def PaymentMethodAgent(self, state:AgentSchema):
        print("Invoking PaymentMethodAgent")
        TEMPLATE = ChatPromptTemplate.from_messages([
            ("system", self.db_prompt.format(dialect=self.paymentmethod_db.dialect, top_k=20, business_id=state["business_id"])),
            MessagesPlaceholder("agent_scratchpad"),
            ("user", "{user_input}")
        ])

        agent = CreateAgent(db=self.paymentmethod_db, llm=self.llm, prompt=TEMPLATE)
    
        response = agent.invoke({"user_input":state["user_query"]})["output"]
    
        return {"agent_response" : response}
    
    def CategoryAndBrandAgent(self, state:AgentSchema):
        print("Invoking CategoryAndBrandAgent")
        TEMPLATE = ChatPromptTemplate.from_messages([
            ("system", self.db_prompt.format(dialect=self.categories_and_brand_db.dialect, top_k=20, business_id=state["business_id"])),
            MessagesPlaceholder("agent_scratchpad"),
            ("user", "{user_input}")
        ])

        agent = CreateAgent(db=self.categories_and_brand_db, llm=self.llm, prompt=TEMPLATE)
    
        response = agent.invoke({"user_input":state["user_query"]})["output"]
    
        return {"agent_response" : response}

    def CompileAgent(self):
        graph = StateGraph(state_schema=self.agent_schema)

        graph.add_node("ConversationAgent", self.ConversationAgent)
        graph.add_node("CustomerSalesAgent", self.CustomerSalesAgent)
        graph.add_node("InventoryAgent", self.InventoryAgent)
        graph.add_node("ExpenseAgent", self.ExpenseAgent)
        graph.add_node("SearchAgent", self.SearchAgent)
        graph.add_node("PointOfSalesAgent", self.PointOfSalesAgent)
        graph.add_node("PaymentMethodAgent", self.PaymentMethodAgent)
        graph.add_node("CategoryAndBrandAgent", self.CategoryAndBrandAgent)

        graph.add_conditional_edges(START,
                                    self.RouteQuery,
                                    {"conversation_agent":"ConversationAgent",
                                     "customer_sales_agent":"CustomerSalesAgent",
                                     "inventory_agent":"InventoryAgent",
                                     "expense_agent":"ExpenseAgent",
                                     "search_agent":"SearchAgent",
                                     "point_of_sales_agent":"PointOfSalesAgent",
                                     "payment_methods_agent":"PaymentMethodAgent",
                                     "category&brand_agent":"CategoryAndBrandAgent"})

        graph.add_edge("ConversationAgent", END)
        graph.add_edge("CustomerSalesAgent", END)
        graph.add_edge("InventoryAgent", END)
        graph.add_edge("ExpenseAgent", END)
        graph.add_edge("SearchAgent", END)
        graph.add_edge("PointOfSalesAgent", END)
        graph.add_edge("PaymentMethodAgent", END)
        graph.add_edge("CategoryAndBrandAgent", END)
        
        return graph.compile()
    
llm = ChatOpenAI(model="gpt-4.1", temperature=0)

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
f"\nprevious_user_query:{state['previous_user_query']}\nprevious_ai_response:{state['previous_ai_response']}\nuser_query:{state['user_query']}"
)
    return state

helfercorps_bot_ed = RunnableLambda(add_context) | helfercorps_bot