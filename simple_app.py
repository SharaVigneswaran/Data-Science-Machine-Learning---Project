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

def display_difficulty(prediction, display_animation):
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
    st.markdown(suggestions[prediction])

############ 5. INTERACTIVE QUIZ ############

# Define quiz questions and answers
quiz_questions = {
    "A1": {
        "question": "Which is the correct conjugation of 'être' for 'nous'?",
        "options": ["es", "suis", "sommes", "êtes"],
        "answer": "sommes"
    },
    "A2": {
        "question": "Translate 'I have a dog' to French.",
        "options": ["Je suis un chien", "J'ai un chien", "Tu as un chien", "Il a un chien"],
        "answer": "J'ai un chien"
    },
    "B1": {
        "question": "Which sentence uses the future tense with 'aller'?",
        "options": ["Je vais manger", "Je mange", "J'ai mangé", "Je mangerai"],
        "answer": "Je vais manger"
    },
    "B2": {
        "question": "Form a complex sentence using the subjunctive present with 'pouvoir'.",
        "options": ["Il faut que tu puisses finir ce projet avant vendredi.", "Je peux finir ce projet.", "Il faut finir ce projet.", "Je finirai ce projet."],
        "answer": "Il faut que tu puisses finir ce projet avant vendredi."
    },
    "C1": {
        "question": "Which sentence discusses an abstract idea?",
        "options": ["La liberté est essentielle pour le développement personnel.", "Je mange une pomme.", "Il fait beau aujourd'hui.", "Elle a un chat noir."],
        "answer": "La liberté est essentielle pour le développement personnel."
    },
    "C2": {
        "question": "Translate 'The juxtaposition of tradition and modernity in French culture is fascinating.' to French.",
        "options": ["La juxtaposition de la tradition et de la modernité dans la culture française est fascinante.", "La culture française est intéressante.", "Il y a une juxtaposition dans la culture française.", "La tradition en France est moderne."],
        "answer": "La juxtaposition de la tradition et de la modernité dans la culture française est fascinante."
    }
}

def display_quiz(level):
    if level in quiz_questions:
        question = quiz_questions[level]["question"]
        options = quiz_questions[level]["options"]
        correct_answer = quiz_questions[level]["answer"]

        st.markdown(f"**Quiz Question:** {question}")
        user_answer = st.radio("Choose an answer:", options, key=f"quiz_{level}")

        if st.button("Submit Answer", key=f"submit_{level}"):
            if user_answer == correct_answer:
                st.success("Correct!")
            else:
                st.error(f"Incorrect! The correct answer is: {correct_answer}")

# Call this function to display the quiz after the difficulty level is determined
def main():
    if 'history' not in st.session_state:
        st.session_state.history = []

    sentence = st.text_input("Enter a sentence to classify its difficulty level:", "")

    if sentence:
        if "last_input" not in st.session_state or sentence != st.session_state.last_input:
            st.session_state.last_input = sentence
            prediction = predict_difficulty(sentence)
            display_difficulty(prediction, display_animation)
            display_quiz(prediction)
            # Update history
            st.session_state.history.append((sentence, prediction))
        else:
            # Retain the previous prediction and display the quiz
            prediction = predict_difficulty(sentence)
            display_quiz(prediction)

    if show_history and st.session_state.history:
        st.write("### Sentence History")
        for sent, pred in reversed(st.session_state.history):
            st.text(f"Sentence: {sent} - Level: {pred}")
            
    st.write("### Enhance Your French Skills")
    st.write("""
        Improving your French doesn't stop here! Here are some tips and partner apps that can help you enhance your French proficiency:

        **Tips:**
        - Practice speaking with native speakers as often as possible.
        - Watch French movies and series with subtitles.
        - Read French books, newspapers, and articles.
        - Write daily journals or essays in French to improve your writing skills.
        - Use flashcards for vocabulary building.

        **Recommended Apps:**
        - **Duolingo:** A fun app for learning languages with gamified lessons.
        - **Babbel:** Offers interactive courses with a focus on real-life conversations.
        - **Memrise:** Helps you learn through spaced repetition and mnemonic techniques.
        - **HelloTalk:** Connects you with native speakers for language exchange.
        - **Tandem:** Another great app for finding language exchange partners.
        - **LingQ:** Provides extensive reading and listening resources to immerse yourself in French.
    """)

############ 6. VOCABULARY BUILDING ############
vocabulary = {
    "A1": [
        {"word": "chat", "definition": "cat", "example": "Le chat dort."},
        {"word": "chien", "definition": "dog", "example": "Le chien aboie."}
    ],
    "A2": [
        {"word": "maison", "definition": "house", "example": "La maison est grande."},
        {"word": "voiture", "definition": "car", "example": "La voiture est rouge."}
    ],
    "B1": [
        {"word": "liberté", "definition": "freedom", "example": "La liberté est essentielle."},
        {"word": "culture", "definition": "culture", "example": "La culture française est riche."}
    ],
    "B2": [
        {"word": "complexe", "definition": "complex", "example": "C'est une idée complexe."},
        {"word": "développement", "definition": "development", "example": "Le développement personnel est important."}
    ],
    "C1": [
        {"word": "nuance", "definition": "nuance", "example": "Il y a une nuance subtile dans son discours."},
        {"word": "réfléchi", "definition": "thoughtful", "example": "C'est un commentaire réfléchi."}
    ],
    "C2": [
        {"word": "juxtaposition", "definition": "juxtaposition", "example": "La juxtaposition des idées est fascinante."},
        {"word": "profondeur", "definition": "depth", "example": "Il parle avec beaucoup de profondeur."}
    ]
}

def display_vocabulary(level):
    if level in vocabulary:
        st.markdown("**Vocabulary List:**")
        for entry in vocabulary[level]:
            word = entry["word"]
            definition = entry["definition"]
            example = entry["example"]
            st.markdown(f"**{word}**: {definition}")
            st.markdown(f"_Example_: {example}")

############ ADDITIONAL VISUAL ELEMENTS ############

# Adding a footer image or branding

if __name__ == "__main__":
    main()
