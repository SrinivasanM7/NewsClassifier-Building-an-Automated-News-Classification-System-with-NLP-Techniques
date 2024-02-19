import streamlit as st
import pickle
import time
import spacy
import re
import pandas as pd

# Setting Webpage Configurations
st.set_page_config(page_title="Automated News Classifier", layout="wide")

st.title(':blue[Automated News Classifier🚀]')

# Loading the model and the Vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Input
text = st.text_input('Enter your Headline')

# Text processing
# Text preprocessing function
def processing(text):
    processing = spacy.load('en_core_web_md')    
    corpus = []

    # Replacing numbers and Special Characters with whitespace
    news = re.sub('[^a-zA-Z\s]', '', text)
    
    # Convert the String to lowercase
    news = news.lower()

    # Removing Stop word and Lemmatisation
    doc = processing(news)
    news = [token.lemma_ for token in doc if not token.is_stop]
    news = ' '.join(news)

    corpus.append({'Header':news})

    processed_header = pd.DataFrame(data = corpus, columns = ['Header'])

    return processed_header

processed_df = processing(text)

# Submit buttonb
submit = st.button('Classify')

if submit:
    with st.spinner('Please wait'):
        time.sleep(1)
     
    input_vectorized = vectorizer.transform(processed_df['Header'])
  
    input_prediction = model.predict(input_vectorized)

    if input_prediction == 0:
        input_prediction = 'Technology'

    elif input_prediction == 1:
        input_prediction = 'Automobile'

    elif input_prediction == 2:
        input_prediction = 'Health and Science'
    
    elif input_prediction == 3:
        input_prediction = 'Investing'
    
    elif input_prediction == 4:
        input_prediction = 'Politics'

    st.subheader(f':green[Section] : {input_prediction}')

    st.success('Done!')
