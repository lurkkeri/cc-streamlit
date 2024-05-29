import streamlit as st
import pandas as pd
import joblib
import sklearn

import nltk

import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.data.path.append('/path/to/your/local/nltk_data')

model = joblib.load('spam_model.pkl')

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Remove non-alphabetic characters and lower the case
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    
    # Tokenize and remove stopwords
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize tokens
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

def predict(input_data):
    # Ensure input_data is in a DataFrame
    input_df = pd.DataFrame([input_data], columns=['text'])
    
    # Perform any necessary preprocessing on the input_data
    input_df['text'] = input_df['text'].apply(preprocess_text)
    
    # Make predictions using the model
    prediction = model.predict(input_df['text'])
    return prediction[0]

# Streamlit app
def main():
    st.title('Streamlit App with Pickle Model')

    # Add input components (e.g., text input, file upload, etc.)
    input_data = st.text_input('Enter text for prediction:')
    
    # Example of making predictions when a button is clicked
    if st.button('Predict'):
        # Ensure input_data is not empty
        if input_data:
            # Call the predict function with input_data
            prediction = predict(input_data)
            result = 'Spam' if prediction == 1 else 'Ham'
            st.write('Prediction:', result)
        else:
            st.write('Please enter text for prediction')

if __name__ == '__main__':
    main()