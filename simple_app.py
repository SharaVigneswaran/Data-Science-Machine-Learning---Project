import streamlit as st

# Import your model and any necessary preprocessing functions
from your_model_module import predict_language_level

# Define the Streamlit app layout
def main():
    st.title("Language Proficiency Classifier")

    # Text input for the user to enter a sentence
    sentence = st.text_input("Enter a sentence:")

    if st.button("Classify"):
        # Call your model to predict the language proficiency level
        if sentence.strip() != "":
            prediction = predict_language_level(sentence)
            st.write("Predicted Language Level:", prediction)
        else:
            st.write("Please enter a sentence.")

if __name__ == "__main__":
    main()
