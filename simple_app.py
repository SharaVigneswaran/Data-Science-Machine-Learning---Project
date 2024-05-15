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
    
# Friendly suggestions for improvement based on level
    suggestions = {
        "A1": ("To move from A1 to A2, try adding more adjectives and basic conjunctions (e.g., et, mais). "
               "Expand your vocabulary with common nouns and verbs. "
               "For example, instead of 'Le chat dort,' you could say 'Le chat noir dort et rêve.'"),
        "A2": ("To move from A2 to B1, start using more complex sentence structures, such as relative clauses. "
               "Increase your use of past and future tenses. "
               "For example, instead of 'Je mange une pomme,' try 'Je mange une pomme que j'ai achetée hier.'"),
        "B1": ("To move from B1 to B2, focus on using more advanced grammar structures, including the subjunctive mood. "
               "Improve your vocabulary with less common words and idiomatic expressions. "
               "For example, instead of 'Je pense que c'est bon,' try 'Il faut que tu saches que c'est excellent.'"),
        "B2": ("To move from B2 to C1, aim to perfect your use of advanced tenses and moods. "
               "Enhance your ability to discuss abstract ideas and complex topics. "
               "For example, instead of 'Je veux voyager,' try 'J'aspire à explorer de nouvelles cultures.'"),
        "C1": ("To move from C1 to C2, work on achieving near-native fluency. "
               "Focus on nuanced language use, including stylistic elements and advanced idiomatic expressions. "
               "For example, instead of 'C'est intéressant,' try 'Cela suscite un intérêt profond et réfléchi.'"),
        "C2": ("Congratulations! You've reached the highest proficiency level. "
               "Continue practicing to maintain and further refine your skills. "
               "Engage in complex discussions and read a variety of French literature to stay sharp.")
    }

    exercises = {
        "A1": ("Construct a sentence with an adjective and a conjunction. For example: 'The big cat and the small dog are friends.'",
               "Le grand chat et le petit chien sont amis."),
        "A2": ("Write a sentence using a relative clause. For example: 'I eat the apple that I bought yesterday.'",
               "Je mange la pomme que j'ai achetée hier."),
        "B1": ("Formulate a sentence using the subjunctive mood. For example: 'It is necessary that you know this.'",
               "Il faut que tu saches cela."),
        "B2": ("Create a sentence discussing an abstract idea. For example: 'I aspire to explore new cultures.'",
               "J'aspire à explorer de nouvelles cultures."),
        "C1": ("Write a nuanced sentence with stylistic elements. For example: 'This provokes deep and thoughtful interest.'",
               "Cela suscite un intérêt profond et réfléchi."),
        "C2": ("Discuss a complex topic in a sentence. For example: 'The juxtaposition of tradition and modernity in French culture is fascinating.'",
               "La juxtaposition de la tradition et de la modernité dans la culture française est fascinante.")
    }

    st.markdown(f"**Suggestion:** {suggestions[prediction]}")

    exercise_text, answer = exercises[prediction]
    st.markdown(f"**Exercise:** {exercise_text}")

    if st.button(f"Show Answer for {prediction}"):
        st.markdown(f"**Answer:** {answer}")
    
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

