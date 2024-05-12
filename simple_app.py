import streamlit as st
from PIL import Image

############ 1. SETTING UP THE PAGE LAYOUT AND TITLE ############

# Configure the Streamlit page with layout settings, title, and icon
st.set_page_config(
    layout="centered", page_title="LogoRank", page_icon="📚"
)

############ 2. CREATE THE LOGO AND HEADING ############

# Using columns to layout the logo and title side by side
c1, c2 = st.columns([0.2, 1.8])

with cols[0]:
    # Displaying a logo image if available
    try:
        st.image(Image.open("Capture d’écran 2024-05-10 à 11.36.38.png"), width=60)  
    except Exception as e:
        st.error(f"Error loading logo: {e}")

with cols[1]:
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

# Main interaction: text input and instant feedback
sentence = st.text_input("Enter a sentence to classify its difficulty level:", "")

# Using Streamlit's session state to manage app states
if sentence:
    if not "last_input" in st.session_state or sentence != st.session_state.last_input:
        st.session_state.last_input = sentence
        prediction = predict_difficulty(sentence)
        display_difficulty(prediction)
