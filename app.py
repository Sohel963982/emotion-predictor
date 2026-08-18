import streamlit as st
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load model and vectorizer
model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
emotion_numbers = joblib.load("emotion_numbers.pkl")

# Reverse dictionary
number_emotion = {v: k for k, v in emotion_numbers.items()}

# Stopwords
stop_word = set(stopwords.words('english'))


# ---------------- PREPROCESSING FUNCTIONS ----------------

def remove_punc(txt):
    return txt.translate(
        str.maketrans('', '', string.punctuation)
    )


def remove_numbers(txt):
    new = ""

    for i in txt:
        if not i.isdigit():
            new = new + i

    return new


def remove_emojis(txt):
    new = ""

    for i in txt:
        if i.isascii():
            new = new + i

    return new


def remove_stopwords(txt):

    words = word_tokenize(txt)

    clean_txt = []

    for i in words:

        if i not in stop_word:
            clean_txt.append(i)

    return ' '.join(clean_txt)


# Complete preprocessing
def preprocess_text(txt):

    txt = txt.lower()

    txt = remove_punc(txt)

    txt = remove_numbers(txt)

    txt = remove_emojis(txt)

    txt = remove_stopwords(txt)

    return txt


# ---------------- STREAMLIT FRONTEND ----------------

st.title("Emotion Classification System 😊")

st.write("Enter a sentence and the model will predict the emotion.")

user_input = st.text_area(
    "Enter your text here:"
)


if st.button("Predict Emotion"):

    if user_input.strip() == "":

        st.warning("Please enter some text.")

    else:

        # Preprocess input
        cleaned_text = preprocess_text(user_input)

        # Convert text into TF-IDF
        input_tfidf = vectorizer.transform([cleaned_text])

        # Prediction
        prediction = model.predict(input_tfidf)

        # Convert number back to emotion name
        predicted_emotion = number_emotion[prediction[0]]

        st.success(
            f"Predicted Emotion: {predicted_emotion.upper()}"
        )

        st.write("### Processed Text")
        st.write(cleaned_text)