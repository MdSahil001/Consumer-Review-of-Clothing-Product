# Placeholder deep learning model
import tensorflow as tf
from tensorflow.keras import layers, models

def create_model(num_labels, vocab_size=20000, embed_dim=128):
    """
    Simple baseline multilabel classification model.
    """

    model = models.Sequential([
        layers.Embedding(vocab_size, embed_dim, input_length=200),
        layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
        layers.GlobalMaxPool1D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_labels, activation="sigmoid")  # multilabel output
    ])

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    return model
