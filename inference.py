import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# Load pre-trained model and tokenizer from the saved model directory
model = RobertaForSequenceClassification.from_pretrained('./results')  # Update path if needed
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def predict(texts):
    # Tokenize input texts
    encodings = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
    
    # Get predictions from the model
    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

    # Map numerical predictions back to sentiment labels
    sentiment_mapping_reverse = {0: 'negative', 1: 'neutral', 2: 'positive'}
    predicted_labels = [sentiment_mapping_reverse[pred.item()] for pred in predictions]
    
    return predicted_labels

# Example usage of the function
if __name__ == "__main__":
    sample_texts = [
        "The flight was ontimes and it was a great  experience.",
        "The crew was very friendly and helpful.",
        "It was an average flight."
    ]
    
    predictions = predict(sample_texts)
    for text, sentiment in zip(sample_texts, predictions):
        print(f"Text: {text}\nPredicted Sentiment: {sentiment}\n")
