import streamlit as st
from PIL import Image
import time

############ 1. SETTING UP THE PAGE LAYOUT AND TITLE ############

# Configure the Streamlit page with layout settings, title, and icon
st.set_page_config(layout="wide", page_title="LogoRank", page_icon="📚")

############ 2. SIDEBAR FOR APP SETTINGS ############

# Sidebar for user settings or additional options
with st.sidebar:
    st.title("Settings")
    display_animation = st.checkbox("Animate Progress Bar", value=True)
    show_history = st.checkbox("Show Sentence History", value=True)

############ 3. MAIN PAGE LAYOUT ############

# Using columns to layout the main components
c1, c2, c3 = st.columns([0.1, 0.8, 0.1])

with c2:
    st.image("images/Logo.jpeg")  # An illustrative image related to language learning
    st.title("Test Your French Level")

############ 4. APP FUNCTIONALITY ############

def predict_difficulty(sentence):
    # Placeholder prediction logic
    words_count = len(sentence.split())
    return "A1" if words_count < 10 else "A2" if words_count < 20 else "B1" if words_count < 30 else "B2" if words_count < 40 else "C1" if words_count < 50 else "C2"

def display_difficulty(prediction):
    difficulty_scale = {'A1': (0.1, '🟢', 'Beginner'), 'A2': (0.2, '🟡', 'Elementary'),
                        'B1': (0.4, '🔵', 'Intermediate'), 'B2': (0.6, '🟣', 'Upper Intermediate'),
                        'C1': (0.8, '🟠', 'Advanced'), 'C2': (1.0, '🔴', 'Proficiency')}
    progress_value, emoji, level_desc = difficulty_scale[prediction]

    if display_animation:
        # Function to animate progress
        with st.empty():
            for percent_complete in range(int(progress_value * 100) + 1):
                time.sleep(0.05)
                st.progress(percent_complete / 100.0)

    st.markdown(f"**Difficulty Level:** {emoji} {prediction} - {level_desc}")

if 'history' not in st.session_state:
    st.session_state.history = []

sentence = st.text_input("Enter a sentence to classify its difficulty level:", "")

if sentence:
    if "last_input" not in st.session_state or sentence != st.session_state.last_input:
        st.session_state.last_input = sentence
        prediction = predict_difficulty(sentence)
        display_difficulty(prediction)
        # Update history
        st.session_state.history.append((sentence, prediction))

if show_history and st.session_state.history:
    st.write("### Sentence History")
    for sent, pred in reversed(st.session_state.history):
        st.text(f"Sentence: {sent} - Level: {pred}")

############ 5. SUGGESTIONS TO MODIFY SENTENCE ############



############ ADDITIONAL VISUAL ELEMENTS ############

# Adding a footer image or branding
st.image("images/Logo.jpeg", caption="Enhance your French with LogoRank", width=300)

