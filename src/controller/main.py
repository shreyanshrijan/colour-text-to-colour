from __future__ import annotations

import pickle
import numpy as np
import tensorflow as tf

from src.service import (
    _create_training_data, vectorize_words, _create_model,
    _create_model_with_Word2Vec, _sample_generate
)


def train_model_end_to_end(model_name: str, epochs: int):
    """Controller function to train the model and save it.
    """

    X, y, P, text_vec, col_words, color_data = _create_training_data()
    colour_prediction_model = _create_model(vocabulary_size=len(col_words))

    y_mean = tf.math.reduce_mean(tf.Variable(y), axis=0)
    y_std = tf.math.reduce_std(tf.Variable(y), axis=0)
    data_generated = _sample_generate(X, y, y_mean, y_std, P)

    init_vals = []
    for ke, va in col_words.items():
        matching_cols = [v for k, v in color_data.items() if ke in k.split()]

        if len(matching_cols) > 0:
            init_vals.append(np.median(np.vstack(matching_cols), axis=0))
        else:
            init_vals.append(np.array([0, 0, 0]))

    # train on the data
    colour_prediction_model.fit(data_generated, steps_per_epoch=100, epochs=epochs)

    # Save the model
    pickle.dump(
        {
            'y_mean': y_mean,
            'y_std': y_std
        },
        open(f"{model_name}_params.pkl", 'wb')
    )
    colour_prediction_model.save(f"{model_name}.h5")
    print("Model saved successfully")


def train_model_end_to_end_with_Word2Vec(model_name: str, epochs: int):
    """Train end to end model using Word2Vec for embedding the words and then using GRU to learn
    the patterns.
    """
    X, y, P, word2vec_model = vectorize_words()

    colour_prediction_model_with_word2vec = _create_model_with_Word2Vec()

    y_mean = np.mean(y, axis=0)
    y_std = np.std(y, axis=0)
    data_generated = _sample_generate(X, y, y_mean, y_std, P)

    colour_prediction_model_with_word2vec.fit(data_generated, epochs=epochs, steps_per_epoch=1000)

    pickle.dump(
        {
            'y_mean': y_mean,
            'y_std': y_std
        },
        open(f"{model_name}_params_with_word2vec.pkl", 'wb')
    )
    colour_prediction_model_with_word2vec.save(f"{model_name}_with_word2vec.h5")
    word2vec_model.save("word2vec_model.model")
    print("Model saved successfully")


if __name__ == '__main__':
    train_model_end_to_end_with_Word2Vec("model_testing", 200)
