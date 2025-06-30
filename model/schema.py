from pydantic import Field
from pydantic import BaseModel
from typing import TypedDict, Literal

class MultiAgentResponseModel(BaseModel):
    agent_response:str

class AgentSchema(TypedDict):
    user_query:str
    agent_response:str
    business_id:int
    previous_user_query:str
    previous_ai_response:str
    
class AgentRouter(BaseModel):
    route: Literal[
        "conversation_agent",
        "customer_sales_agent",
        "inventory_agent",
        "expense_agent",
        "search_agent", 
        "point_of_sales_agent",
        "payment_methods_agent",
        "category&brand_agent",
        "analysis_agent"
    ] = Field(
        ..., 
        description=(
            "Specifies the type of agent to route the query to. "
            "Options include: 'conversation_agent' for general dialogue, "
            "'customer_sales_agent' for questions related to customer sales, "
            "'inventory_agent' for inventory analysis, "
            "'expense_agent' for expense tracking and management, "
            "'search_agent' for price recommendations, "
            "'point_of_sales_agent' for queries related to point of sales and walk-in sales, "
            "'payment_methods_agent' for queries related to payment methods, "
            "'category&brand_agent' for queries related to selling categories and selling brands."
            "'analysis_agent' To aggregate and analyze data from various agents, generating actionable insights, performance summaries, and trend reports for decision-making."
        )
    )
    
class AnalystSchema(TypedDict):
    user_query:str
    bot_response:str