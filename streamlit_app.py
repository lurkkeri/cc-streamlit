import re
import zipfile
import streamlit as st
import pandas as pd
import joblib
import nltk
import os
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Set NLTK data path
nltk_data_path = '/cc-streamlit/nltk_data/'
#nltk_data_path = 'C:\\Users\\Oona\\AppData\\Roaming\\nltk_data'
nltk.data.path.append(nltk_data_path)


required_directories = ['corpora/stopwords', 'corpora/wordnet', 'tokenizers/punkt']

missing_directories = []
for d in required_directories:
    try:
        nltk.data.find(d)
    except LookupError:
        missing_directories.append(d)

print("NLTK data path:", nltk.data.path)
print("Missing directories:", missing_directories)

def unzip_wordnet(zip_path, extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

# Check if wordnet is unzipped, if not, unzip it
wordnet_zip_path = os.path.join(nltk_data_path, 'corpora/wordnet.zip')
wordnet_extract_path = os.path.join(nltk_data_path, 'corpora/wordnet')
if not os.path.exists(wordnet_extract_path):
    unzip_wordnet(wordnet_zip_path, wordnet_extract_path)

if missing_directories:
    for directory in missing_directories:
        resource_name = directory.split('/')[-1]
        nltk.download(resource_name, download_dir=nltk_data_path)



# Load the pickled model
model = joblib.load('spam_model.pkl')

# Define preprocessing function
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

# Define a function to make predictions using the loaded model
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
