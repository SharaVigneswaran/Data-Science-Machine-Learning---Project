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

def display_difficulty(prediction):
    difficulty_scale = {'A1': (0.1, '👶'), 'A2': (0.2, '🧒'), 'B1': (0.4, '👦'), 'B2': (0.6, '🧑'), 'C1': (0.8, '👨'), 'C2': (1.0, '🧓')}
    progress_value, emoji = difficulty_scale[prediction]
    st.progress(progress_value)
    st.markdown(f"**Difficulty Level:** {emoji} {prediction}")

def main():
    st.title("Language Proficiency Classifier")

    sentence = st.text_input("Enter a sentence:")
    if st.button("Classify"):
        if sentence.strip() != "":
            prediction = predict_difficulty(sentence)
            display_difficulty(prediction)
        else:
            st.warning("Please enter a sentence to classify its difficulty level.")

if __name__ == "__main__":
    main()
