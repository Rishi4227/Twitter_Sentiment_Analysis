# %% 
# Import necessary libraries
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# %% 
# Load the trained model and tokenizer
start_time = time.time()
model = RobertaForSequenceClassification.from_pretrained('./results')
tokenizer = RobertaTokenizer.from_pretrained('./results')
end_time = time.time()
print(f"Model and tokenizer loaded in {end_time - start_time:.2f} seconds")

# %% 
# Load the validation dataset
start_time = time.time()
print("Loading the validation dataset...")
dataset = pd.read_csv('Tweets.csv')  # Make sure to point to the right dataset
sampled_dataset = dataset.sample(frac=0.3, random_state=42)  # Sample 30% of the dataset

# Convert the true labels to numeric form to match the predicted labels format
sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
sampled_dataset['airline_sentiment'] = sampled_dataset['airline_sentiment'].map(sentiment_mapping)

validation_texts = sampled_dataset['text'].tolist()
true_labels = sampled_dataset['airline_sentiment'].tolist()
end_time = time.time()
print(f"Validation dataset loaded in {end_time - start_time:.2f} seconds")

# %% 
# Tokenize the validation texts with proper padding and truncation
start_time = time.time()
print("Tokenizing validation texts...")
val_encodings = tokenizer(validation_texts, truncation=True, padding=True, return_tensors="pt")
end_time = time.time()
print(f"Tokenization complete in {end_time - start_time:.2f} seconds")

# %% 
# Run predictions in batches with progress monitoring
start_time = time.time()
print("Running predictions in batches...")

batch_size = 32
pred_labels = []
model.eval()

for i in tqdm(range(0, len(val_encodings['input_ids']), batch_size), desc="Running predictions in batches"):
    batch = {k: v[i:i+batch_size] for k, v in val_encodings.items()}
    with torch.no_grad():
        outputs = model(**batch)
        logits = outputs.logits
        batch_preds = torch.argmax(logits, dim=1).tolist()
        pred_labels.extend(batch_preds)

end_time = time.time()
print(f"Predictions completed in {end_time - start_time:.2f} seconds")

# %% 
# Calculate metrics and display classification report
print("Calculating metrics and generating reports...")
report = classification_report(true_labels, pred_labels, target_names=['negative', 'neutral', 'positive'])
print(report)

# %% 
# Confusion Matrix
print("Generating confusion matrix...")
cm = confusion_matrix(true_labels, pred_labels)
cmd = ConfusionMatrixDisplay(cm, display_labels=['negative', 'neutral', 'positive'])
cmd.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# %% 
# Plot Precision, Recall, F1-score
print("Plotting Precision, Recall, and F1-Score...")
report_df = pd.DataFrame(classification_report(true_labels, pred_labels, output_dict=True)).transpose()

plt.figure(figsize=(10, 5))
sns.heatmap(report_df.iloc[:-1, :-1].T, annot=True, cmap='Blues')
plt.title("Precision, Recall, and F1-Score")
plt.show()

# %% 
# Plot Loss Curves (using fake data for demonstration)
epochs = list(range(1, 4))  # Replace this with the actual number of epochs from training
train_loss = [0.6, 0.4, 0.3]  # Replace with actual training loss from the model
val_loss = [0.7, 0.5, 0.4]  # Replace with actual validation loss

plt.figure(figsize=(10, 5))
plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# %% 
# Plot Accuracy Curves (using fake data for demonstration)
train_acc = [0.75, 0.85, 0.9]  # Replace with actual training accuracy from the model
val_acc = [0.7, 0.8, 0.85]  # Replace with actual validation accuracy

plt.figure(figsize=(10, 5))
plt.plot(epochs, train_acc, label='Train Accuracy')
plt.plot(epochs, val_acc, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# %% 
# Advanced Plotly Visualization for interactive plots (optional)
fig = go.Figure()

fig.add_trace(go.Scatter(x=epochs, y=train_loss, mode='lines', name='Train Loss'))
fig.add_trace(go.Scatter(x=epochs, y=val_loss, mode='lines', name='Val Loss'))

fig.update_layout(title='Training and Validation Loss',
                  xaxis_title='Epochs',
                  yaxis_title='Loss')

fig.show()

