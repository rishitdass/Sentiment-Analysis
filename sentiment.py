import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize
stopwords=stopwords.words("english")
stemmer=PorterStemmer()

#preprocessing
def transform_text(text):
    text = re.sub('[^a-zA-Z1-9\s]', ' ', text.lower())

    text = nltk.word_tokenize(text)
    x = []

    for i in text:
        if i not in stopwords and i not in string.punctuation:
            x.append(i)

    text = x.copy()
    x.clear()

    for i in text:
        x.append(stemmer.stem(i))

    return " ".join(x)
tfidf = pickle.load(open('sentiment_vectorizer.pkl','rb'))
model = pickle.load(open('sentiment_model.pkl','rb'))

st.title("Sentiment analysis Classifier with Summarization3")

input_sms = st.text_area("Enter the review")
import requests

#Text summarizer
#enter you api key from hugging face
API_TOKEN="#eneter your api key from hugging face"
headers = {"Authorization": f"Bearer {API_TOKEN}"}
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()
data = query(
    {
        "inputs": input_sms,
        "parameters": {"do_sample": False},
    }
)

#pridict part
if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.markdown(
            f'<h1 style="color:blue;">Review seems to be Positive</h1>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<h1 style="color:red;">Review seems to be Negative</h1>',
            unsafe_allow_html=True
        )
    st.markdown(
        f'<h1 style="color:green; text-decoration: underline;">Summary-</h1>',
        unsafe_allow_html=True
    )
    sentences = sent_tokenize(data[0]["summary_text"])
    for sentence in sentences:
        st.write(sentence)