from __future__ import annotations

import pickle
import keras
import tensorflow as tf
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


# @elementwise
def predict_on_test_data(x_test: list, text_vector, model, y_mean, y_std):

    return model.predict(text_vector.call(x_test)) * y_std + y_mean


def predict(model_name: str, colour_name: str):

    try:
        colour_prediction_model = keras.models.load_model(f"{model_name}.h5")

        # Load the text vector data from pickle file
        from_disk = pickle.load(open(f"{model_name}_params.pkl", 'rb'))

    except NoModelFound:
        print("Training a new model with the user input name")  # Bring logging in the pipeline

        train_model_end_to_end(model_name)

        colour_prediction_model = keras.models.load_model(f"{model_name}.h5")

        # Load the text vector data from pickle file
        from_disk = pickle.load(open(f"{model_name}_params.pkl", 'rb'))

    text_vector = keras.layers.TextVectorization(
        output_mode='int', output_sequence_length=from_disk['config']['output_sequence_length']
    )

    # text_vector = keras.layers.TextVectorization.from_config(from_disk['config'])
    # You have to call `adapt` with some dummy data (BUG in Keras)
    # text_vector.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
    text_vector.set_weights(from_disk['weights'])
    y_mean = from_disk['y_mean']
    y_std = from_disk['y_std']
    print(text_vector.get_vocabulary())
    print(y_mean)
    print(y_std)
    # predicted_colour = predict_on_test_data(
    #     [colour_name], text_vector, colour_prediction_model, y_mean, y_std
    # )
    predicted_colour = colour_prediction_model.predict(
        text_vector.call([colour_name])
    ) * y_std + y_mean

    return draw_color_palletes(predicted_colour.numpy())


if __name__ == '__main__':
    predict("model testing", "dark red")
