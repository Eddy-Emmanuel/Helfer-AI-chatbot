from fastapi import FastAPI
from routes import multi_agent

app = FastAPI(root_path="/ai")

app.include_router(router=multi_agent.router)

