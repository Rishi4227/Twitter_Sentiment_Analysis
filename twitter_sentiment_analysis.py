# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.metrics import accuracy_score, f1_score
import os

# Display current working directory
print("Current Working Directory:", os.getcwd())

# Load and preprocess the Airline Twitter Sentiment dataset
print("Loading Airline Twitter Sentiment dataset...")
dataset = pd.read_csv('Tweets.csv')  # Update with the correct path to the new dataset

# Check the distribution of airline_sentiment
print(dataset['airline_sentiment'].value_counts())

# Keep only necessary columns
dataset = dataset[['airline_sentiment', 'text']]

# Convert sentiment to numerical labels (0 for negative, 1 for neutral, 2 for positive)
sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
dataset['airline_sentiment'] = dataset['airline_sentiment'].map(sentiment_mapping)
print("Dataset loaded and preprocessed.")

# Take a smaller sample of the dataset (adjusted size to 50%)
sampled_dataset = dataset.sample(frac=0.5, random_state=42)

# Split the smaller dataset into training and validation sets (80% training, 20% validation)
print("Splitting the data...")
train_texts, val_texts, train_labels, val_labels = train_test_split(
    sampled_dataset['text'], sampled_dataset['airline_sentiment'], test_size=0.2, random_state=42
)
print(f"Training set size: {len(train_texts)}, Validation set size: {len(val_texts)}")

# %% Tokenize Dataset
print("Loading RoBERTa tokenizer...")
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Tokenize training and validation texts
train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True)
val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True)
print("Tokenization complete.")

# %% Create PyTorch Dataset
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Create train and validation datasets
train_dataset = SentimentDataset(train_encodings, train_labels.tolist())
val_dataset = SentimentDataset(val_encodings, val_labels.tolist())
print("Datasets created.")

# Define compute_metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = torch.tensor(logits)  # Convert logits to tensors
    predictions = torch.argmax(logits, dim=-1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')  # Weighted for multi-class
    return {"accuracy": accuracy, "f1": f1}

# %% Load Model and Train
print("Loading RoBERTa model...")
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)  # For 3 classes

# Training arguments - optimized for performance
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=6,  # Increase epochs for better training
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    gradient_accumulation_steps=1,
    evaluation_strategy="epoch",
    learning_rate=2e-5,  # Use a lower learning rate
    weight_decay=0.01,  # Add weight decay to prevent overfitting
    logging_dir='./logs',
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,  # Include metrics computation
)

# Start training
print("Starting training...")
trainer.train()
print("Training complete.")

# Save the trained model and tokenizer
model.save_pretrained('./results')
tokenizer.save_pretrained('./results')

print("Model and tokenizer saved.")











# # Import necessary libraries
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
# import torch
# from sklearn.metrics import accuracy_score, f1_score
# import os

# # Display current working directory
# print("Current Working Directory:", os.getcwd())

# # Load and preprocess the Airline Twitter Sentiment dataset
# print("Loading Airline Twitter Sentiment dataset...")
# dataset = pd.read_csv('Tweets.csv')  # Update with the correct path to the new dataset
# # Check the distribution of airline_sentiment
# print(dataset['airline_sentiment'].value_counts())
# dataset = dataset[['airline_sentiment', 'text']]  # Keep only necessary columns

# # Convert sentiment to numerical labels (0 for negative, 1 for neutral, 2 for positive)
# sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
# dataset['airline_sentiment'] = dataset['airline_sentiment'].map(sentiment_mapping)
# print("Dataset loaded and preprocessed.")

# # Take a smaller sample of the dataset (adjusted size)
# sampled_dataset = dataset.sample(frac=0.5, random_state=42)  # Increased sample size to 30%

# # Split the smaller dataset into training and validation sets (80% training, 20% validation)
# print("Splitting the data...")
# train_texts, val_texts, train_labels, val_labels = train_test_split(
#     sampled_dataset['text'], sampled_dataset['airline_sentiment'], test_size=0.2, random_state=42
# )
# print(f"Training set size: {len(train_texts)}, Validation set size: {len(val_texts)}")

# # %% Tokenize Dataset
# print("Loading RoBERTa tokenizer...")
# tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# # Tokenize training and validation texts
# train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True)
# val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True)
# print("Tokenization complete.")

# # %% Create PyTorch Dataset
# class SentimentDataset(torch.utils.data.Dataset):
#     def __init__(self, encodings, labels):
#         self.encodings = encodings
#         self.labels = labels

#     def __getitem__(self, idx):
#         item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
#         item['labels'] = torch.tensor(self.labels[idx])
#         return item

#     def __len__(self):
#         return len(self.labels)

# # Create train and validation datasets
# train_dataset = SentimentDataset(train_encodings, train_labels.tolist())
# val_dataset = SentimentDataset(val_encodings, val_labels.tolist())
# print("Datasets created.")

# # Define compute_metrics function
# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     logits = torch.tensor(logits)  # Convert logits to tensors
#     predictions = torch.argmax(logits, dim=-1)
#     accuracy = accuracy_score(labels, predictions)
#     f1 = f1_score(labels, predictions, average='weighted')  # Weighted for multi-class
#     return {"accuracy": accuracy, "f1": f1}

# # %% Load Model and Train
# print("Loading RoBERTa model...")
# model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)  # For 3 classes

# # Training arguments - optimized for performance
# training_args = TrainingArguments(
#     output_dir='./results',
#     num_train_epochs=3,  # Increase epochs for better training
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=64,
#     gradient_accumulation_steps=1,
#     evaluation_strategy="epoch",
#     learning_rate=2e-5,  # Use a lower learning rate
#     weight_decay=0.01,  # Add weight decay to prevent overfitting
#     logging_dir='./logs',
#     logging_steps=10,
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
#     compute_metrics=compute_metrics,  # Include metrics computation
# )

# print("Starting training...")
# trainer.train()
# print("Training complete.")

# # Starting training...
# trainer.train()
# print("Training complete.")

# # Save the trained model and tokenizer
# model.save_pretrained('./results')
# tokenizer.save_pretrained('./results')
