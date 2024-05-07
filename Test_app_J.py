import streamlit as st
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import CamembertTokenizer, CamembertForSequenceClassification

# Function to load the model
@st.cache(allow_output_mutation=True)
def load_model():
    model = CamembertForSequenceClassification.from_pretrained('camembert-base', num_labels=len(label_encoder.classes_))
    model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))  # Load your trained model
    model.eval()
    return model

# Function to make predictions
def predict_difficulty(text):
    inputs = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_label_idx = torch.argmax(logits, dim=1).item()
    predicted_label = label_encoder.inverse_transform([predicted_label_idx])[0]
    return predicted_label

# Load label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = pd.read_csv('label_encoder_classes.csv')['class'].values  # Load your label encoder classes from a CSV

# Load tokenizer
tokenizer = CamembertTokenizer.from_pretrained('camembert-base')

# Load model
model = load_model()

# Streamlit app
st.title('Text Difficulty Classification')

text_input = st.text_area("Enter a sentence:", "")
if st.button("Classify"):
    if text_input.strip() == "":
        st.error("Please enter a sentence.")
    else:
        predicted_label = predict_difficulty(text_input)
        st.success(f"Predicted Difficulty: {predicted_label}")
