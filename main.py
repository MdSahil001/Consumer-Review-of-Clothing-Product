# Placeholder main script for training and evaluating
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import tensorflow as tf

from model import create_model

DATA_PATH = "data/sample.csv"

def load_dataset():
    df = pd.read_csv(DATA_PATH)

    # First column = review text
    texts = df.iloc[:, 0].astype(str).tolist()

    # Remaining columns = labels (1/0)
    label_cols = df.columns[1:]
    labels = df[label_cols].values

    return texts, labels, label_cols


def preprocess_text(texts, vocab_size=20000, max_len=200):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_len)

    return padded, tokenizer


def main():
    print("Loading dataset...")
    texts, labels, label_names = load_dataset()

    print("Preprocessing text...")
    X, tokenizer = preprocess_text(texts)

    print(f"Detected {labels.shape[1]} labels.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42
    )

    print("Creating model...")
    model = create_model(num_labels=labels.shape[1])

    print("Training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=5,
        batch_size=32
    )

    print("Saving model...")
    model.save("saved_model/")

    print("Training complete!")
    print("Model saved in ./saved_model/")


if __name__ == "__main__":
    main()
