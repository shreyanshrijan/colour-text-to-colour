from __future__ import annotations

import pickle
import numpy as np
import tensorflow as tf

from src.service import _create_training_data, _create_model, _sample_generate


def train_model_end_to_end(model_name: str):
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
    colour_prediction_model.fit(data_generated, steps_per_epoch=3200, epochs=200)

    # Save the model
    pickle.dump(
        {
            'config': text_vec.get_config(),
            'weights': text_vec.get_weights(),
            'y_mean': y_mean,
            'y_std': y_std
        },
        open(f"{model_name}_params.pkl", 'wb')
    )
    colour_prediction_model.save(f"{model_name}.h5")
    print("Model saved successfully")