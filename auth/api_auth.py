from fastapi import status
from fastapi import Security
from typing import Annotated
from fastapi.security import APIKeyHeader
from fastapi.exceptions import HTTPException
from config.secret_keys import project_config

apikeyheader = APIKeyHeader(name="api_auth_key")

async def authenticate_key(api_key:Annotated[str, Security(apikeyheader)]):
    if api_key == project_config.api_key:
        return api_key
    else:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, 
                            detail="Could not validate API-key")
        
