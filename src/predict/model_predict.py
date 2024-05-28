from __future__ import annotations

import pickle
import keras
import tensorflow as tf
from functools import wraps

from src.common.exceptions import NoModelFound
from src.controller.main import train_model_end_to_end
from src.inference.test_performance import draw_color_palletes
from src.service import _create_training_data

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


# @elementwise
def predict_on_test_data(x_test: list, text_vector, model, y_mean, y_std):

    return model.predict(text_vector.call(x_test)) * y_std + y_mean


def predict(model_name: str, colour_name: str):

    X, y, P, text_vec, col_words, color_data = _create_training_data()
    try:
        colour_prediction_model = keras.models.load_model(f"{model_name}.h5")

    except NoModelFound:
        print("Training a new model with the user input name")  # Bring logging in the pipeline

        train_model_end_to_end(model_name)

        colour_prediction_model = keras.models.load_model(f"{model_name}.h5")

    # Load the text vector data from pickle file
    from_disk = pickle.load(open(f"{model_name}_params.pkl", 'rb'))

    y_mean = from_disk['y_mean']
    y_std = from_disk['y_std']

    predicted_colour = colour_prediction_model.predict(text_vec.call([colour_name])) * y_std + y_mean
    print("***************************************")
    print(predicted_colour)

    return draw_color_palletes(predicted_colour.numpy())


if __name__ == '__main__':
    predict("test_shreyan", "dark red")
