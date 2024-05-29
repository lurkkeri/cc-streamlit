import streamlit as st
import pandas as pd
import joblib

# Load the pickled model
model = joblib.load('spam_model.pkl')

# Define a function to make predictions using the loaded model
def predict(input_data):
    # Perform any necessary preprocessing on the input_data
    # Make predictions using the model
    prediction = model.predict(input_data)
    return prediction

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
            st.write('Prediction:', prediction)
        else:
            st.write('Please enter text for prediction')

if __name__ == '__main__':
    main()