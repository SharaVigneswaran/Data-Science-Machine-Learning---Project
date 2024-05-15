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
c1, c2, c3 = st.columns([0.2, 0.6, 0.2])

with c2:
    st.image("images/Logo.jpeg", use_column_width=True)  # An illustrative image related to language learning
    st.title("Test Your French Level")
    st.write("""
        Welcome to LogoRank, where language learning meets innovation! Our vision at LogoRank is to revolutionize the way you learn French. With cutting-edge technology and a passion for education, we're dedicated to enhancing your language learning experience like never before.

        Join us on a journey where you can progress at your own pace, empowering you to reach new heights in your French proficiency. By integrating LogoRank into your daily learning routine, you'll unlock the key to mastering French effortlessly.

        Simply type your sentence below, and let LogoRank determine your current CEFR level in French. Start your language learning adventure with us today!
    """)

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

def conjugate_verb(verb, tense="présent"):
    conjugator = Conjugator()
    conjugated_forms = conjugator.conjugate(verb)[tense]
    return conjugated_forms

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
st.image("images/Logo.jpeg", width=600, caption="LogoRank")
