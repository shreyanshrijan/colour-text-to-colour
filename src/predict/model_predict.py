from __future__ import annotations

import json
import pickle
import keras
from collections import ChainMap
import pandas as pd
from functools import wraps

from src.common.exceptions import NoModelFound
from src.controller.main import train_model_end_to_end
from src.inference.test_performance import draw_color_palletes

def elementwise(func):
    @wraps(func)
    def wrapper(self, input, *args, **kwargs):
        if isinstance(input, list):
            output = []
            for input_i in input:
                output.append(wrapper(self, input_i, *args, **kwargs))
        else:
            output = func(self, input, *args, **kwargs)
        return output

    return wrapper


@elementwise
def predict_on_test_data(x_test: list, text_vector, model, y_mean, y_std):

    return model.predict(text_vector.call(x_test)) * y_std + y_mean


def predict(model_name: str, colour_name: str):

    try:
        colour_prediction_model = keras.models.load_model(f"{model_name}.h5")

        # Load the text vector data from pickle file
        from_disk = pickle.load(open(f"{model_name}_params.pkl", 'rb'))

    except NoModelFound:
        print("Training a new model with the user input name")  # Bring lof=gging in the pipeline

        train_model_end_to_end(model_name)

        colour_prediction_model = keras.models.load_model(f"{model_name}.h5")

        # Load the text vector data from pickle file
        from_disk = pickle.load(open(f"{model_name}_params.pkl", 'rb'))

    text_vector = keras.layers.TextVectorization.from_config(from_disk['config'])
    text_vector.set_weights(from_disk['weights'])
    y_mean = from_disk['y_mean']
    y_std = from_disk['y_std']

    predicted_colour = predict_on_test_data(
        [colour_name], text_vector, colour_prediction_model, y_mean, y_std
    )

    return draw_color_palletes(predicted_colour.numpy())