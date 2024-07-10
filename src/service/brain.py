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


def _create_model_with_Word2Vec() -> keras.Model:

    colour_prediction_model = keras.models.Sequential([
        keras.layers.Masking(mask_value=0.0, input_shape=(6, 5)),
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
        optimizer=keras.optimizers.legacy.Adam(learning_rate=0.01),
        loss='mean_squared_error'
    )

    return colour_prediction_model
