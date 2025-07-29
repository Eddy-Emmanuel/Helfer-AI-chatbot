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
        "helfer_db",
        "search_agent"
    ] = Field(
        ..., 
        description=(
            "Specifies the type of agent to route the query to. "
            "Options include: 'conversation_agent' for general dialogue, "
            "'helfer_db' for databse related queries, "
            "'search_agent' for price recommendations, "
        )
    )
    
class AnalystSchema(TypedDict):
    user_query:str
    bot_response:str