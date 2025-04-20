from fastapi import FastAPI
from routes import basic

app = FastAPI()

app.include_router(router=basic.route)