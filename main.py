import uvicorn
from fastapi import FastAPI
from routes import multi_agent

app = FastAPI(root_path="/ai")

app.include_router(router=multi_agent.router)

if __name__ == "__main__":
    uvicorn.run(app=app, port=8000, host="0.0.0.0")
    # uvicorn.run(app=app)