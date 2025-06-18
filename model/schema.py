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
        "sales_agent",
        "inventory_agent",
        "expense_agent",
        "search_agent", 

    ] = Field(
        ..., 
        description=(
            "Specifies the type of agent to route the query to. "
            "Options include: 'conversation_agent' for general dialogue, "
            "'sales_agent' for sales-related queries, "
            "'inventory_agent' for inventory analysis, "
            "'expense_agent' for expense tracking and management, "
            "'search_agent' for price recommendations."  
        )
    )
    
class AnalystSchema(TypedDict):
    user_query:str
    bot_response:str