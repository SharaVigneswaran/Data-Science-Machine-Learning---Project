import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import CamembertTokenizer, CamembertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class FrenchTextDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}  # Use clone().detach()
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def prepare_data(texts, labels, tokenizer):
    text_list = texts.tolist()  # Convert Series to list
    text_list = [str(text) for text in text_list]  # Ensure each element is a string
    encodings = tokenizer(text_list, truncation=True, padding=True, max_length=128, return_tensors="pt")
    dataset = FrenchTextDataset(encodings, labels)
    return dataset

def main():
    # Load your data here
    df_training_data = pd.read_csv('your_data.csv')  # Replace with your actual data loading

    # Label encoding
    label_encoder = LabelEncoder()
    df_training_data['encoded_labels'] = label_encoder.fit_transform(df_training_data['difficulty'])

    # Tokenizer
    tokenizer = CamembertTokenizer.from_pretrained('camembert-base')

    # Split data and prepare datasets
    train_texts, val_texts, train_labels, val_labels = train_test_split(df_training_data['sentence'], df_training_data['encoded_labels'], test_size=0.1, random_state=42)
    train_dataset = prepare_data(train_texts, train_labels.tolist(), tokenizer)
    val_dataset = prepare_data(val_texts, val_labels.tolist(), tokenizer)

    # Model
    model = CamembertForSequenceClassification.from_pretrained('camembert-base', num_labels=len(label_encoder.classes_))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Optimizer and scheduler setup
    optimizer = AdamW(model.parameters(), lr=3e-5)
    total_steps = len(train_dataset) * 5  # Assuming 5 epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Training loop
    model.train()
    for epoch in range(5):
        for batch in DataLoader(train_dataset, batch_size=16, shuffle=True):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    # Save the model
    model.save_pretrained('camembert_model')
    tokenizer.save_pretrained('camembert_tokenizer')

    # Evaluation
    model.eval()
    val_loader = DataLoader(val_dataset, batch_size=16)
    val_preds, val_labels = [], []
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        val_preds.extend(predictions.cpu().numpy())
        val_labels.extend(batch['labels'].cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(val_labels, val_preds)
    precision, recall, fscore, _ = precision_recall_fscore_support(val_labels, val_preds, average='weighted')

    print(f'Validation Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {fscore}')

if __name__ == '__main__':
    main()
