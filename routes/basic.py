import pandas as pd
from typing import Annotated
from io import StringIO, BytesIO
from utils.bot_utils import AnalyzeData
from config.schema import Tier_1_Response
from auth.api_auth import authenticate_key
from fastapi.responses import JSONResponse
from fastapi import Depends, APIRouter, UploadFile, File, Form, status, HTTPException
import logging

logger = logging.getLogger(__name__)

route = APIRouter(prefix="/routes", tags=["tier 1"])

@route.post(path="/tier_1", response_model=Tier_1_Response)
async def tier_1(
    api_key: Annotated[str, Depends(authenticate_key)],
    file: Annotated[UploadFile, File(..., description="Upload CSV or Excel file.")],
    prompt: Annotated[str, Form(..., description="Prompt for analysis")]
):
    if not api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key.")

    file_content = await file.read()
    ext = file.filename.lower().split(".")[-1]

    try:
        if ext == "csv":
            data = pd.read_csv(StringIO(file_content.decode()))
        elif ext in {"xls", "xlsx"}:
            sheets_dict = pd.read_excel(BytesIO(file_content), sheet_name=None)
            data = pd.concat([df.assign(SheetName=name) for name, df in sheets_dict.items()], ignore_index=True)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported file type. Only csv, xlsx, and xls are allowed."
            )
            
        bot_response = await AnalyzeData(data=data, user_prompt=prompt)

        return JSONResponse(status_code=status.HTTP_200_OK, content={"response":bot_response})
    
    except Exception as e:
        logger.exception("Error processing uploaded file")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to analyze file: {str(e)}"
        )
