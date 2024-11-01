import torch
import pandas as pd
import matplotlib.pyplot as plt
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_validation_data(filepath):
    dataset = pd.read_csv(filepath)
    texts = dataset['text'].values
    labels = dataset['airline_sentiment'].map({'negative': 0, 'neutral': 1, 'positive': 2}).values
    return texts, labels

def evaluate_checkpoint(checkpoint_path, tokenizer, validation_texts, validation_labels):
    # Load model
    print(f"Loading model from checkpoint: {checkpoint_path}")
    model = RobertaForSequenceClassification.from_pretrained(checkpoint_path)
    model.eval()

    # Tokenize inputs
    print(f"Tokenizing data for checkpoint: {checkpoint_path}")
    encodings = tokenizer(validation_texts.tolist(), truncation=True, padding=True, max_length=128, return_tensors='pt')

    # Extract input IDs and attention mask
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']

    # Run model and get predictions
    print(f"Running model for checkpoint: {checkpoint_path}")
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=-1)

    # Calculate metrics
    accuracy = accuracy_score(validation_labels, predictions)
    f1 = f1_score(validation_labels, predictions, average='weighted')
    return accuracy, f1

def main():
    print("Loading tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    validation_texts, validation_labels = load_validation_data('Tweets.csv')  # Update with correct path

    checkpoint_paths = [
        'results/checkpoint-74', 'results/checkpoint-500', 
        'results/checkpoint-660', 'results/checkpoint-1000', 
        'results/checkpoint-1500', 'results/checkpoint-2000', 
        'results/checkpoint-2196'
    ]

    accuracies, f1_scores = [], []

    for checkpoint in checkpoint_paths:
        print(f"Evaluating checkpoint {checkpoint}")
        accuracy, f1 = evaluate_checkpoint(checkpoint, tokenizer, validation_texts, validation_labels)
        accuracies.append(accuracy)
        f1_scores.append(f1)
        print(f"Checkpoint {checkpoint}: Accuracy = {accuracy:.4f}, F1 Score = {f1:.4f}")

    # Plotting the results
    print("Plotting results...")
    plt.figure(figsize=(10, 5))
    plt.plot(checkpoint_paths, accuracies, label='Accuracy', marker='o')
    plt.plot(checkpoint_paths, f1_scores, label='F1 Score', marker='o')
    plt.xlabel('Checkpoints')
    plt.ylabel('Score')
    plt.title('Model Performance over Checkpoints')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Starting evaluation...")
    main()
    print("Evaluation complete.")
