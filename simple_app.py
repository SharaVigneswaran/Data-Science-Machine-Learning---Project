import streamlit as st

def predict_difficulty(sentence):
    # Ideally, you would use your trained model here to predict the difficulty level
    # For now, let's just return a placeholder prediction based on the length of the sentence
    if len(sentence.split()) < 10:
        return "A1"
    elif len(sentence.split()) < 20:
        return "A2"
    elif len(sentence.split()) < 30:
        return "B1"
    elif len(sentence.split()) < 40:
        return "B2"
    elif len(sentence.split()) < 50:
        return "C1"
    else:
        return "C2"

def main():
    st.title("Language Proficiency Classifier")

    # Text input for the user to enter a sentence
    sentence = st.text_input("Enter a sentence:")

    if st.button("Classify"):
        # Predict the difficulty level based on the input sentence
        if sentence.strip() != "":
            prediction = predict_difficulty(sentence)
            st.write("Predicted Difficulty Level:", prediction)
            # Add visual representation of difficulty level
            difficulty_color = {'A1': 'red', 'A2': 'orange', 'B1': 'yellow', 'B2': 'green', 'C1': 'blue', 'C2': 'purple'}
            st.markdown(f"**Difficulty Level:** <font color='{difficulty_color[prediction]}'>{prediction}</font>", unsafe_allow_html=True)
        else:
            st.write("Please enter a sentence.")

if __name__ == "__main__":
    main()
