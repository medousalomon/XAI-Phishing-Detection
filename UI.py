import streamlit as st
import pickle
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from lime.lime_text import LimeTextExplainer
import sys
import os

# Fix import path
sys.path.append(os.path.abspath("../src"))
from Preprocess import clean_text


# LOAD MODEL + TOKENIZER

model = tf.keras.models.load_model("../models/phishing_model.h5")

with open("../models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)



# PREDICTION FUNCTION

def predict_email(text):
    cleaned = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=200, padding='post')

    prediction = model.predict(padded)[0][0]

    return prediction



# LIME EXPLANATION

def explain_email(text):
    explainer = LimeTextExplainer(class_names=["Legitimate", "Phishing"])

    def predict_proba(texts):
        cleaned = [clean_text(t) for t in texts]
        seq = tokenizer.texts_to_sequences(cleaned)
        padded = pad_sequences(seq, maxlen=200, padding='post')
        preds = model.predict(padded)
        return np.hstack((1 - preds, preds))

    explanation = explainer.explain_instance(text, predict_proba, num_features=10)

    return explanation.as_list()



# STREAMLIT UI

st.title("📧 Phishing Email Detection System")

email_text = st.text_area("Paste Email Text Here")

if st.button("Detect"):
    if email_text.strip() == "":
        st.warning("Please enter an email.")
    else:
        prediction = predict_email(email_text)

        if prediction > 0.5:
            st.error(f"⚠️ PHISHING ({prediction:.4f})")
        else:
            st.success(f"✅ LEGITIMATE ({1 - prediction:.4f})")

        st.subheader("🔍 Explanation")

        explanation = explain_email(email_text)

        for word, weight in explanation:
            st.write(f"{word}: {weight:.4f}")