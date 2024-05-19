import streamlit as st
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
import time

############ 1. SETTING UP THE PAGE LAYOUT AND TITLE ############

# Configure the Streamlit page with layout settings, title, and icon
st.set_page_config(layout="wide", page_title="LogoRank")

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
        Welcome to LogoRank! We're here to help you quickly find out your current French level and give you the tools to keep improving.

        LogoRank is an innovative app developed by two EPFL students, designed to revolutionize the way you learn French. Using advanced CamemBERT machine learning technology, LogoRank assesses your CEFR level in French with just a simple sentence input.

        Just type in a sentence below, and LogoRank will provide you with an accurate level assessment. We'll also offer personalized tips and resources to help you on your language learning journey. Let's get started and make learning French fun and easy!
    """)

############ 4. APP FUNCTIONALITY ############
@st.cache_resource
def load_model_and_tokenizer():
    model_path = "saved_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, from_safetensors=True)
    return model, tokenizer

def predict_difficulty(sentence, model, tokenizer):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    label_map = {0: 'A1', 1: 'A2', 2: 'B1', 3: 'B2', 4: 'C1', 5: 'C2'}
    return label_map[prediction]

def display_difficulty(prediction, display_animation):
    difficulty_scale = {
        'A1': (0.1, '🟢', 'Beginner'), 
        'A2': (0.2, '🟡', 'Elementary'),
        'B1': (0.4, '🔵', 'Intermediate'), 
        'B2': (0.6, '🟣', 'Upper Intermediate'),
        'C1': (0.8, '🟠', 'Advanced'), 
        'C2': (1.0, '🔴', 'Proficiency')
    }
    progress_value, emoji, level_desc = difficulty_scale[prediction]

    if display_animation:
        progress_bar = st.progress(0)
        for percent_complete in range(int(progress_value * 100) + 1):
            time.sleep(0.05)
            progress_bar.progress(percent_complete / 100.0)

    st.markdown(f"**Difficulty Level:** {emoji} {prediction} - {level_desc}")

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
    "A1": [
        {
            "question": "Which is the correct conjugation of 'être' for 'nous'?",
            "options": ["es", "suis", "sommes", "êtes"],
            "answer": "sommes"
        }, 
        {
            "question": "Translate 'cat' to French.",
            "options": ["chien", "chat", "oiseau", "poisson"],
            "answer": "chat"
        },
        {
            "question": "Which is the correct article for 'chien'?",
            "options": ["le", "la", "les", "un"],
            "answer": "le"
        }
    ],
    "A2": [
        {
            "question": "Translate 'I have a dog' to French.",
            "options": ["Je suis un chien", "J'ai un chien", "Tu as un chien", "Il a un chien"],
            "answer": "J'ai un chien"
        },
        {
            "question": "Which is the correct conjugation of 'avoir' for 'vous' in passé composé?",
            "options": ["avez eu", "a eu", "avais eu", "auriez eu"],
            "answer": "avez eu"
        },
        {
            "question": "Translate 'house' to French.",
            "options": ["maison", "voiture", "école", "église"],
            "answer": "maison"
        }
    ],
    "B1": [
        {
            "question": "Which sentence uses the future tense with 'aller'?",
            "options": ["Je vais manger", "Je mange", "J'ai mangé", "Je mangerai"],
            "answer": "Je vais manger"
        },
        {
            "question": "Translate 'We are going to the park' to French.",
            "options": ["Nous allons au parc", "Nous allons à l'école", "Nous sommes au parc", "Nous allons manger"],
            "answer": "Nous allons au parc"
        },
        {
            "question": "Which is the correct conjugation of 'faire' for 'ils' in futur simple?",
            "options": ["ferons", "ferez", "feront", "ferait"],
            "answer": "feront"
        }
    ],
    "B2": [
        {
            "question": "Form a complex sentence using the subjunctive present with 'pouvoir'.",
            "options": ["Il faut que tu puisses finir ce projet avant vendredi.", "Je peux finir ce projet.", "Il faut finir ce projet.", "Je finirai ce projet."],
            "answer": "Il faut que tu puisses finir ce projet avant vendredi."
        },
        {
            "question": "Translate 'I want you to come with me' to French.",
            "options": ["Je veux que tu viennes avec moi.", "Je veux que tu viens avec moi.", "Je veux que tu viendras avec moi.", "Je veux que tu es venu avec moi."],
            "answer": "Je veux que tu viennes avec moi."
        },
        {
            "question": "Which is the correct conjugation of 'aller' for 'nous' in subjonctif présent?",
            "options": ["allions", "allons", "alliez", "aille"],
            "answer": "allions"
        }
    ],
    "C1": [
        {
            "question": "Which sentence discusses an abstract idea?",
            "options": ["La liberté est essentielle pour le développement personnel.", "Je mange une pomme.", "Il fait beau aujourd'hui.", "Elle a un chat noir."],
            "answer": "La liberté est essentielle pour le développement personnel."
        },
        {
            "question": "Translate 'He speaks fluently and confidently' to French.",
            "options": ["Il parle couramment et avec confiance.", "Il parle lentement et timidement.", "Il parle vite et fort.", "Il parle doucement et gentiment."],
            "answer": "Il parle couramment et avec confiance."
        },
        {
            "question": "Which is the correct conjugation of 'venir' for 'je' in plus-que-parfait?",
            "options": ["étais venu(e)", "étais venu", "étais venue", "étais venir"],
            "answer": "étais venu(e)"
        }
    ],
    "C2": [
        {
            "question": "Translate 'The juxtaposition of tradition and modernity in French culture is fascinating.' to French.",
            "options": ["La juxtaposition de la tradition et de la modernité dans la culture française est fascinante.", "La culture française est intéressante.", "Il y a une juxtaposition dans la culture française.", "La tradition en France est moderne."],
            "answer": "La juxtaposition de la tradition et de la modernité dans la culture française est fascinante."
        },
        {
            "question": "Which sentence uses a complex structure correctly?",
            "options": ["Bien que fatigué, il a continué à travailler.", "Parce que fatigué, il a continué à travailler.", "Même fatigué, il a continué à travailler.", "Fatigué, il a continué à travailler."],
            "answer": "Bien que fatigué, il a continué à travailler."
        },
        {
            "question": "Which is the correct conjugation of 'savoir' for 'je' in conditionnel passé?",
            "options": ["aurais su", "aurais sais", "aurais sait", "aurais savoir"],
            "answer": "aurais su"
        }
    ]
}

def display_quiz(level):
    if level in quiz_questions:
        questions = quiz_questions[level]
        
        for i, q in enumerate(questions):
            col1, col2 = st.columns([0.6, 0.4])  # Adjust the column width as needed
            
            with col1:
                question = q["question"]
                options = q["options"]
                correct_answer = q["answer"]

                st.markdown(f"**Quiz Question {i+1}:** {question}")
                user_answer = st.radio("Choose an answer:", options, key=f"quiz_{level}_{i}")

                if st.button(f"Submit Answer {i+1}", key=f"submit_{level}_{i}"):
                    if user_answer == correct_answer:
                        st.success("Correct!")
                    else:
                        st.error(f"Incorrect! The correct answer is: {correct_answer}")

            if i == 1:
                with col2:
                    st.image("images/Learning French.jpeg", use_column_width=True)  

############ MAIN FUNCTION ############
def main():
    if 'history' not in st.session_state:
        st.session_state.history = []

    sentence = st.text_input("Enter a sentence to classify its difficulty level:", "")

    if sentence:
        model, tokenizer = load_model_and_tokenizer()
        if "last_input" not in st.session_state or sentence != st.session_state.last_input:
            st.session_state.last_input = sentence
            prediction = predict_difficulty(sentence, model, tokenizer)
            display_difficulty(prediction, display_animation)
            
            st.write("### Now let's test your knowledge further with a quick quiz!")
            display_quiz(prediction)
            st.session_state.history.append((sentence, prediction))
        else:
            prediction = st.session_state.history[-1][1]
            st.write("### Now let's test your knowledge with a quick quiz!")
            display_quiz(prediction)

    if show_history and st.session_state.history:
        st.write("### Check Your Progress")
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
    """)
    
    # Add a related video
    

    st.write("""
        **Recommended Apps:**
        - **[Duolingo](https://fr.duolingo.com/):** A fun app for learning languages with gamified lessons.
        - **[Babbel](https://fr.babbel.com/):** Offers interactive courses with a focus on real-life conversations.
        - **[Memrise](https://www.memrise.com/fr/):** Helps you learn through spaced repetition and mnemonic techniques.
        - **[HelloTalk](https://www.hellotalk.com/):** Connects you with native speakers for language exchange.
        - **[Tandem](https://www.tandem.net/fr):** Another great app for finding language exchange partners.
        - **[LingQ](https://www.lingq.com/fr/):** Provides extensive reading and listening resources to immerse yourself in French.
    """)


############ ADDITIONAL VISUAL ELEMENTS ############

# Adding a footer image or branding
# st.image("images/footer_image.jpg", use_column_width=True)

if __name__ == "__main__":
    main()
