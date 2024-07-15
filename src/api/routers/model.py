from __future__ import annotations

from fastapi import APIRouter, Response, BackgroundTasks

from src.api.dto.request import ModelTraining, ModelInference
from src.controller.main import train_model_end_to_end
from src.predict.model_predict import predict, predict_word2vec_model


routers = APIRouter(
    prefix="/model",
    tags=["model"],
    responses={404: {"descrription": "not found"}}
)


@routers.get("/")
def get_main():
    return {"msg": "Hello World!"}


@routers.post("/model_train")
def model_training(request: ModelTraining):  # Add the response as well
    train_model_end_to_end(request.colour_model_id, request.epochs)


@routers.post("/model_inference")
def model_prediction(request: ModelInference):
    img_buf, _ = predict_word2vec_model(request.colour_model_id, request.colour_name)
    return Response(img_buf.getvalue(), media_type='image/png')
