!pip install scikit-learn
import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# Function to predict difficulty level based on input sentence
def predict_difficulty(sentence):
    # Load the trained logistic regression model and TF-IDF vectorizer
    logistic_regression_classifier = LogisticRegression(random_state=52, max_iter=10000)
    vectorizer = TfidfVectorizer(max_features=10000)
    
    # Load the training data (replace df_training_data with your actual training data)
    df_training_data = pd.read_csv("training_data.csv")
    X = vectorizer.fit_transform(df_training_data['sentence']).toarray()
    y = df_training_data['difficulty']
    
    # Train the logistic regression classifier
    logistic_regression_classifier.fit(X, y)
    
    # Vectorize the input sentence
    X_sentence = vectorizer.transform([sentence]).toarray()
    
    # Predict the difficulty level
    predicted_difficulty = logistic_regression_classifier.predict(X_sentence)[0]
    
    return predicted_difficulty

# Define the Streamlit app layout
def main():
    st.title("Language Proficiency Classifier (Logistic Regression)")

    # Text input for the user to enter a sentence
    sentence = st.text_input("Enter a sentence:")

    if st.button("Classify"):
        # Predict the difficulty level based on the input sentence
        if sentence.strip() != "":
            prediction = predict_difficulty(sentence)
            st.write("Predicted Difficulty Level:", prediction)
        else:
            st.write("Please enter a sentence.")

if __name__ == "__main__":
    main()
