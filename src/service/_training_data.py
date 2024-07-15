from __future__ import annotations

import json
import unicodedata
from gensim.models import Word2Vec
import numpy as np
import tensorflow.compat.v1.keras.backend as K
import tensorflow as tf
import keras
from collections import Counter
from keras.preprocessing.sequence import pad_sequences

from src.utils import ColourConverter

tf.compat.v1.disable_eager_execution()


def _create_training_data():
    # Get the new word count with only the valid colors
    with open(
        '/Users/shreyanshrijan/Documents/colour-text-to-colour/src/train_test_data/universal_data.json'  # noqa
    ) as f:
        data = f.read()
    data = json.loads(data)

    stop_words = ['a', 'of', 'the', 'an', 'by', 'in', 'at', 'to', 'n', 'no']

    color_data = {strip_accents(
        data[i]['name']
    ).lower(): data[i]['hex'] for i in range(len(data))}

    for k, v in color_data.items():
        color_data[k] = ColourConverter.hex_to_lab(color_data[k])

    col_words = Counter()
    for color_i in color_data.keys():
        col_words.update(color_i.split())

    color_score = lambda color: (
        sum(col_words[w] for w in color.split() if w not in stop_words)
        / sum(1.0 for w in color.split() if w not in stop_words)
    )
    color_data = {k: v for k, v in color_data.items() if color_score(k) > 2}

    col_words = Counter()
    for color_i in color_data.keys():
        col_words.update(w for w in color_i.split() if w not in stop_words)

    X = tf.reshape(tf.constant(list(color_data.keys())), [-1, 1])
    y = tf.constant(list(color_data.values()))

    text_vec = keras.layers.TextVectorization(
        # max_tokens=100,
        output_mode='int',
        output_sequence_length=4
    )

    text_vec.adapt(X)
    X = text_vec.call(X)

    # --------------------
    # Compute the "probability" of selecting a color when training NN
    num_words = sum(col_words.values())

    # define the probability using the same method as word2vec
    def P(z, sub_sample=0.001):
        if z == 0:
            return 0
        else:
            return min((np.sqrt(z / sub_sample) + 1) * sub_sample / z, 1)

    train_prob = np.array([P(
        sum(col_words[w] / num_words for w in color.split() if w in col_words)
    ) for color in color_data.keys()])

    return X, y, train_prob, text_vec, col_words, color_data


def _sample_generate(X, y, y_mean, y_std, P, batch_size=500):

    while True:
        sel_i = np.random.randint(0, X.shape[0], batch_size * 4)
        valid_i = P[sel_i] > np.random.rand(batch_size * 4)

        while valid_i.sum() < batch_size:
            sel_ii = np.random.randint(0, X.shape[0], batch_size * 4)
            sel_i = np.vstack((sel_i, sel_ii))
            valid_i = np.vstack(
                (valid_i, P[sel_ii] > np.random.rand(batch_size * 4))
            )

        x_batch = X[sel_i[valid_i][:batch_size]]
        y_batch = y[sel_i[valid_i][:batch_size]]

        # if augment:
        #     y_batch += np.random.randn(y_batch.shape[0],y_batch.shape[1]) * 0.5

        y_batch = (y_batch - y_mean) / y_std

        yield x_batch, y_batch


# tf.enable_eager_execution() ( Not required for tensorflow version > 2)
def strip_accents(text):
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore')
    text = text.decode("utf-8")
    return str(text)


###################################################################################################
# First Pass in creating Word2Vec
def vectorize_words():
    with open(
        '/Users/shreyanshrijan/Documents/colour-text-to-colour/src/train_test_data/universal_data.json'  # noqa
    ) as f:
        data = f.read()
    data = json.loads(data)

    stop_words = ['a', 'of', 'the', 'an', 'by', 'in', 'at', 'to', 'n', 'no']

    color_data = {strip_accents(
        data[i]['name']
    ).lower(): data[i]['hex'] for i in range(len(data))}

    for k, v in color_data.items():
        color_data[k] = ColourConverter.hex_to_lab(color_data[k])

    col_words = Counter()
    for color_i in color_data.keys():
        col_words.update(color_i.split())

    color_score = lambda color: (
        sum(col_words[w] for w in color.split() if w not in stop_words)
        / sum(1.0 for w in color.split() if w not in stop_words)
    )
    color_data = {k: v for k, v in color_data.items() if color_score(k) > 2}

    col_words = Counter()
    for color_i in color_data.keys():
        col_words.update(w for w in color_i.split() if w not in stop_words)

    tokenized_sentences = [sentence.lower().split() for sentence in list(color_data.keys())]
    word2vec_model = Word2Vec(
        sentences=tokenized_sentences,
        vector_size=50,
        window=3,
        min_count=1,
        workers=4
    )

    def sentence_to_embedding(sentence, model):
        return np.array([model.wv[word] for word in sentence if word in model.wv])

    embedded_sentences = [
        sentence_to_embedding(sentence, word2vec_model) for sentence in tokenized_sentences
    ]

    # max_length = max(len(sentence) for sentence in embedded_sentences)
    max_length = 5
    padded_sentences = pad_sequences(
        embedded_sentences, maxlen=max_length, dtype='float32', padding='post', value=0.0
    )

    y = np.array([v for k, v in color_data.items()])

    # --------------------
    # Compute the "probability" of selecting a color when training NN
    num_words = sum(col_words.values())

    # define the probability using the same method as word2vec
    def P(z, sub_sample=0.001):
        if z == 0:
            return 0
        else:
            return min((np.sqrt(z / sub_sample) + 1) * sub_sample / z, 1)

    train_prob = np.array([P(
        sum(col_words[w] / num_words for w in color.split() if w in col_words)
    ) for color in color_data.keys()])

    return padded_sentences, y, train_prob, word2vec_model


if __name__ == '__main__':
    word_vectors = vectorize_words()
    print(len(word_vectors))
