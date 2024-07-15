from __future__ import annotations

import keras
from keras import regularizers


def _create_model(vocabulary_size) -> keras.Model:

    colour_prediction_model = keras.models.Sequential([
        keras.layers.Embedding(
            vocabulary_size + 1, 128, input_length=None, mask_zero=True
        ),
        keras.layers.GRU(
            128, activation='tanh', return_sequences=True
        ),
        keras.layers.GRU(
            128, activation='tanh', kernel_regularizer=regularizers.l2(0.001)
        ),
        keras.layers.Dense(3, name='rgb_output')
    ])

    colour_prediction_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='mean_squared_error'
    )

    return colour_prediction_model


def _create_model_with_Word2Vec():

    inputs = keras.Input(shape=(5, 50))
    colour_prediction_model = keras.models.Sequential([
        inputs,
        keras.layers.GRU(
            128, return_sequences=True, activation='tanh'
        ),
        keras.layers.GRU(
            128, return_sequences=True, activation='tanh'
        ),
        keras.layers.GRU(
            128, activation='tanh', kernel_regularizer=regularizers.l2(0.01)
        ),
        keras.layers.Dense(3, name='rgb_output')
    ])

    colour_prediction_model.compile(
        optimizer=keras.optimizers.legacy.Adam(learning_rate=0.001),
        loss='mean_squared_error'
    )
    print(colour_prediction_model.summary())

    return colour_prediction_model
