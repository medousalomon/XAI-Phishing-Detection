import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Preprocess import clean_text



# LOAD MODEL + TOKENIZER

model = tf.keras.models.load_model("../models/phishing_model.h5")

with open("../models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)



# PREPROCESS INPUT (LIGHT VERSION)

def preprocess_input(text):
    # Minimal cleaning (same logic as training would be ideal)
    text = clean_text(text)

    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=200, padding='post')

    return padded



# PREDICT FUNCTION

def predict_email(text):
    processed = preprocess_input(text)

    prediction = model.predict(processed)[0][0]

    if prediction > 0.5:
        return "PHISHING", prediction
    else:
        return "LEGITIMATE", prediction



# MAIN (TEST)

if __name__ == "__main__":
    print("Enter an email (type 'exit' to quit):")

    while True:
        user_input = input("\nEmail: ")

        if user_input.lower() == "exit":
            break

        label, score = predict_email(user_input)

        print(f"\nResult: {label}")
        print(f"Confidence: {score:.4f}")