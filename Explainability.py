import pickle
import numpy as np
import tensorflow as tf

from lime.lime_text import LimeTextExplainer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Preprocess import clean_text


# LOAD MODEL + TOKENIZER

model = tf.keras.models.load_model("../models/phishing_model.h5")

with open("../models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)



# PREDICTION FUNCTION FOR LIME

def predict_proba(texts):
    processed = [clean_text(t) for t in texts]
    sequences = tokenizer.texts_to_sequences(processed)
    padded = pad_sequences(sequences, maxlen=200, padding='post')

    preds = model.predict(padded)

    # LIME expects probabilities for both classes
    return np.hstack((1 - preds, preds))



# EXPLAIN FUNCTION

def explain_email(text):
    explainer = LimeTextExplainer(class_names=["Legitimate", "Phishing"])

    explanation = explainer.explain_instance(
        text,
        predict_proba,
        num_features=10
    )

    print("\nExplanation:")
    for word, weight in explanation.as_list():
        print(f"{word}: {weight:.4f}")



# TEST

if __name__ == "__main__":
    sample = input("Enter email to explain:\n")

    explain_email(sample)