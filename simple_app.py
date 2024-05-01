import streamlit as st

# Define a simple function to classify language proficiency based on sentence length
def predict_language_level(sentence):
    # Define thresholds for different proficiency levels
    thresholds = {
        "A1": 5,
        "A2": 10,
        "B1": 15,
        "B2": 20,
        "C1": 25,
        "C2": float('inf')  # Assume any sentence longer than C1 is C2
    }
    
    # Determine the length of the sentence
    sentence_length = len(sentence.split())
    
    # Classify the language proficiency level based on sentence length
    for level, threshold in thresholds.items():
        if sentence_length <= threshold:
            return level

    return "Unknown"  # If sentence length exceeds all thresholds

# Define the Streamlit app layout
def main():
    st.title("Language Proficiency Classifier (Baseline)")

    # Text input for the user to enter a sentence
    sentence = st.text_input("Enter a sentence:")

    if st.button("Classify"):
        # Classify the language proficiency level based on sentence length
        if sentence.strip() != "":
            prediction = predict_language_level(sentence)
            st.write("Predicted Language Level:", prediction)
        else:
            st.write("Please enter a sentence.")

if __name__ == "__main__":
    main()
