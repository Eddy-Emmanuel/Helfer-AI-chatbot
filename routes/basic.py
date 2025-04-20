import pandas as pd
from typing import Annotated
from io import StringIO, BytesIO
from config.schema import Tier_1_Response
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse
from fastapi import APIRouter, UploadFile, File, Form, status

route = APIRouter(prefix="/routes", tags=["tier 1"])

@route.post(path="/tier_1", response_model=Tier_1_Response)
async def tier_1(file:Annotated[UploadFile, 
                                File(..., description="Upload CSV file.")],
                 prompt:Annotated[str, 
                                  Form(..., description="Prompt for analysis")]):
    file_content = await file.read()
    try:
        if file.filename.endswith(".csv"):
            data = pd.read_csv(StringIO(file_content.decode()))
            
        elif file.filename.endswith(".xls") or file.filename.endswith(".xlsx"):
            data = pd.read_excel(BytesIO(file_content))
        
        else:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, 
                                detail="Unsupported file type. Only csv, xlsx, and xls are allowed.")
            
        return JSONResponse(status_code=status.HTTP_200_OK, 
                            content={"response":"Bot response"})
        
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Failed to analyze csv: {str(e)}")
    
    