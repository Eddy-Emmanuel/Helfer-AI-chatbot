from pydantic import BaseModel
from typing import TypedDict, Literal

class AgentSchema(TypedDict):
    user_query:str
    agent_response:str

class AgentRouter(BaseModel):
    route: Literal["conversation_agent", "sales_agent", "inventory_agent", "expense_agent"] = Field(
        ..., 
        description=("Specifies the type of agent to route the query to. "
                     "Options include: 'conversation_agent' for general dialogue, "
                     "'sales_agent' for sales-related queries, 'inventory_agent' for " 
                     "inventory analysis, and 'expense_agent' for expense tracking and management."
                    ))