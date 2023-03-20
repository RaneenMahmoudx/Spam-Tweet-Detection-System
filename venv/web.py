import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
nltk.download('all')
from nltk.stem.porter import PorterStemmer
import base64

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('C:/Users/user/PycharmProjects/ai-2/venv/vectorizer.pkl','rb'))
model = pickle.load(open('C:/Users/user/PycharmProjects/ai-2/venv/model.pkl','rb'))

main_bg = "C:/Users/user/PycharmProjects/ai-2/venv/img_16.png"
main_bg_ext = "png"



st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
    }}
    [theme]
    base="light"

     primaryColor="#FF4B4B"
     backgroundColor="#ffffff"
     secondaryBackgroundColor="#f0f2f6"
     Textcolor = "#31333F"
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style='text-align:center;color:white';>Tweet Spam Classifier</h3>",unsafe_allow_html=True)
input_sms = st.text_area("",value='Enter a Tweet')


if st.button('Predict'):

        # 1. preprocess
        transformed_sms = transform_text(input_sms)
        # 2. vectorize
        vector_input = tfidf.transform([transformed_sms])
        # 3. predict
        result = model.predict(vector_input)[0]
        # 4. Display
        if result == 1:
            st.markdown("<h2 style='text-align:center;color:white';>Spam</h2>",unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='text-align:center;color:white';>Not Spam</h2>",unsafe_allow_html=True)