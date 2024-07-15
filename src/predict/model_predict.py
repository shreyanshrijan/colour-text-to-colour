from __future__ import annotations

import numpy as np
import pickle
import keras
from functools import wraps
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences

from src.common.exceptions import NoModelFound
from src.controller.main import train_model_end_to_end, train_model_end_to_end_with_Word2Vec
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
def predict_on_test_data(x_test: list, text_vector, model: keras.Model, y_mean, y_std):

    return model.predict(text_vector.call(x_test)) * y_std + y_mean


def predict(model_name: str, colour_name: str):

    X, y, P, text_vec, col_words, color_data = _create_training_data()
    try:
        colour_prediction_model: keras.Model = keras.models.load_model(f"{model_name}.h5")

    except NoModelFound:
        print("Training a new model with the user input name")  # Bring logging in the pipeline

        train_model_end_to_end(model_name)

        colour_prediction_model: keras.Model = keras.models.load_model(f"{model_name}.h5")

    # Load the text vector data from pickle file
    from_disk = pickle.load(open(f"{model_name}_params.pkl", 'rb'))

    y_mean = from_disk['y_mean']
    y_std = from_disk['y_std']

    predicted_colour = colour_prediction_model.predict(
        text_vec.call([colour_name])
    ) * y_std + y_mean
    print("***************************************")
    print(predicted_colour)

    return draw_color_palletes(predicted_colour.numpy())


def predict_word2vec_model(model_name: str, colour_name: list[str]):

    try:
        colour_prediction_model: keras.Model = keras.models.load_model(
            f"{model_name}_with_word2vec.h5"
        )

    except NoModelFound:
        print("Training a new model with the user input name")  # Bring logging in the pipeline

        train_model_end_to_end_with_Word2Vec(model_name, 10)

        colour_prediction_model: keras.Model = keras.models.load_model(
            f"{model_name}_with_word2vec.h5"
        )

    word2vec_model: Word2Vec = Word2Vec.load("word2vec_model.model")
    tokenized_sentences = [sentence.lower().split() for sentence in colour_name]

    def sentence_to_embedding(sentence, model: Word2Vec):
        return np.array([model.wv[word] for word in sentence if word in model.wv])

    embedded_sentences = [
        sentence_to_embedding(sentence, word2vec_model) for sentence in tokenized_sentences
    ]

    padded_sentences = pad_sequences(
        embedded_sentences, maxlen=5, dtype='float32', padding='post', value=0.0
    )

    # Load the text vector data from pickle file
    from_disk = pickle.load(open(f"{model_name}_params_with_word2vec.pkl", 'rb'))

    y_mean = from_disk['y_mean']
    y_std = from_disk['y_std']

    predicted_colour = colour_prediction_model.predict(
        padded_sentences
    ) * y_std + y_mean
    print("***************************************")
    print(predicted_colour)

    return draw_color_palletes(predicted_colour)


if __name__ == '__main__':
    # predict("test_shreyan", "dark red")
    print(predict_word2vec_model("model_testing", ["light green"]))
