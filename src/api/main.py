from __future__ import annotations
import uvicorn
from fastapi import FastAPI, Request

from src.api.routers import model


app = FastAPI()
app.include_router(model.routers)


@app.get("/")
def get_main(request: Request):
    return {"message": "Hello world!", "root_path": request.scope.get("root_path")}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info", reload=True)
