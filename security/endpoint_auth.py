from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from config.secret_keys import project_config

key_header = APIKeyHeader(name="x-key", auto_error=False)

def EndPoint_Auth(api_key: str = Security(key_header)):
    if api_key == project_config.endpoint_api_key:
        return api_key
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Invalid or missing API Key"
    )
