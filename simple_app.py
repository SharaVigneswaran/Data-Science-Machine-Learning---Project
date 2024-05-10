import streamlit as st

def predict_difficulty(sentence):
    # Placeholder prediction based on the length of the sentence
    words_count = len(sentence.split())
    if words_count < 10:
        return "A1"
    elif words_count < 20:
        return "A2"
    elif words_count < 30:
        return "B1"
    elif words_count < 40:
        return "B2"
    elif words_count < 50:
        return "C1"
    else:
        return "C2"

def display_difficulty(prediction):
    difficulty_scale = {
        'A1': (0.1, '👶', 'Beginner'),
        'A2': (0.2, '🧒', 'Elementary'),
        'B1': (0.4, '👦', 'Intermediate'),
        'B2': (0.6, '🧑', 'Upper Intermediate'),
        'C1': (0.8, '👨', 'Advanced'),
        'C2': (1.0, '🧓', 'Proficiency')
    }
    progress_value, emoji, level_desc = difficulty_scale[prediction]
    st.progress(progress_value)
    st.markdown(f"**Difficulty Level:** {emoji} {prediction} - {level_desc}")

def main():
    st.set_page_config(page_title="Language Proficiency Classifier", layout="wide")
    st.title("Language Proficiency Classifier")
    st.markdown("### Enter a sentence to classify its difficulty level")

    sentence = st.text_input("", placeholder="Type here...")
    if sentence:
        prediction = predict_difficulty(sentence)
        display_difficulty(prediction)
    else:
        st.warning("Please enter a sentence to classify its difficulty level.")

if __name__ == "__main__":
    main()
