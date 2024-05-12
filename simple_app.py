import streamlit as st
from PIL import Image
import time 

############ 1. SETTING UP THE PAGE LAYOUT AND TITLE ############

# Configure the Streamlit page with layout settings, title, and icon
st.set_page_config(
    layout="centered", page_title="LogoRank", page_icon="📚"
)

############ 2. CREATE THE LOGO AND HEADING ############
# Using columns to layout the logo and title side by side
c1, c2 = st.columns([0.2, 1.8])

with c2:
    # Heading of the app
    st.title("Language Proficiency Classifier")
############ 3. APP FUNCTIONALITY ############

# Placeholder function to predict difficulty based on sentence length
def predict_difficulty(sentence):
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

# Function to display difficulty level with emoji and description
difficulty_scale = {
    'A1': (0.1, '🟢', 'Beginner'),
    'A2': (0.2, '🟡', 'Elementary'),
    'B1': (0.4, '🔵', 'Intermediate'),
    'B2': (0.6, '🟣', 'Upper Intermediate'),
    'C1': (0.8, '🟠', 'Advanced'),
    'C2': (1.0, '🔴', 'Proficiency')
}
    progress_value, emoji, level_desc = difficulty_scale[prediction]

    # Function to animate progress
    def animate_progress(level):
        with st.empty():
            for percent_complete in range(int(level * 100) + 1):  # +1 to ensure it completes
                time.sleep(0.05)
                st.progress(percent_complete / 100.0)

    # Animate the progress bar to the appropriate level
    animate_progress(progress_value)

    # Display the difficulty level after the progress bar animation
    st.markdown(f"**Difficulty Level:** {emoji} {prediction} - {level_desc}")

# History tracking
if 'history' not in st.session_state:
    st.session_state.history = []

# Main interaction: text input and instant feedback
sentence = st.text_input("Enter a sentence to classify its difficulty level:", "")

if sentence:
    if not "last_input" in st.session_state or sentence != st.session_state.last_input:
        st.session_state.last_input = sentence
        prediction = predict_difficulty(sentence)
        display_difficulty(prediction)
        # Update history
        st.session_state.history.append((sentence, prediction))
        # Display history
        st.write("### Sentence History")
        for sent, pred in reversed(st.session_state.history):
            st.text(f"Sentence: {sent} - Level: {pred}")

# Suggestions to modify the sentence
if sentence:
    st.write("### Suggestions to Adjust Difficulty")
    words = sentence.split()
    if len(words) < 10:
        st.markdown("* Try adding more descriptive words or a subordinate clause to increase complexity.")
    elif len(words) > 50:
        st.markdown("* Consider simplifying the sentence by removing adjectives or splitting into two sentences.")
