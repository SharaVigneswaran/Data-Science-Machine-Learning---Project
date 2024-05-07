import random 
import streamlit as st

def predict_difficulty(sentence):
    # Define difficulty levels
    difficulty_levels = ["A1", "A2", "B1", "B2", "C1", "C2"]
    
    # Return a random difficulty level from the list
    return random.choice(difficulty_levels)

# Define the Streamlit app layout
def main():
    st.title("Language Proficiency Classifier (Random Model)")

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
