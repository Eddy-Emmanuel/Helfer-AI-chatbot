import logging
from typing import Annotated, Optional
from security.endpoint_auth import EndPoint_Auth
from model.schema import MultiAgentResponseModel
from utils.ocr_agent_helper_function import LoadFile
from utils.multiagent_helper_function import helfercorps_bot_ed
from fastapi import APIRouter, Depends, HTTPException, status, Body, UploadFile, File

router = APIRouter(
    prefix="/multi_agent", 
    tags=["multi-agent"]
)

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@router.post(path="/", response_model=MultiAgentResponseModel)
async def QueryMultiAgent(
    api_key: Annotated[str, Depends(EndPoint_Auth)],
    user_query:Annotated[str, Body(..., description="User's input query")],
    business_id:Annotated[int, Body(..., description="Business ID per company")],
    previous_user_query: Annotated[str, Body(..., description="Previous user query")] = "No previous user_query",
    previous_ai_response: Annotated[str, Body(..., description="Previous AI response")] = "No previous ai response",
    upload_file:Optional[UploadFile]= None
):
    if not api_key:
        logger.warning("Unauthorized request: Missing or invalid API key.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized"
        )
    
    if upload_file is None:
        response = await helfercorps_bot_ed.ainvoke({"user_query":user_query, 
                                                     "business_id":business_id,
                                                     "previous_user_query":previous_user_query,
                                                     "previous_ai_response":previous_ai_response})
    
    else:       
        loaded_file = LoadFile(file=upload_file)
        
    return MultiAgentResponseModel(agent_response=response["agent_response"])
