# Multilingual BERT Model for Text Classification

import os
from datetime import datetime
import time

import pandas as pd
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score

from textclean import cleancolumn
from trainsplit import splitandbalance

# Model and training parameters
mn = 'mBERT'
data_dir = './classification/data/'

bs = 64  # Batch size
epochs = 6  # Number of epochs
lr = 1e-5  # Learning rate

# Load and preprocess data
input = pd.read_parquet(os.path.join(data_dir, 'goldstandard-dataset.parquet'))
input = cleancolumn(input, 'Text')  # Clean the text column

# Split and balance the dataset into training and validation sets
train, val = splitandbalance(input)

# Initialize the tokenizer for multilingual BERT (large model)
tokenizer = BertTokenizer.from_pretrained('bert-large-multilingual-cased')

# Tokenize the text data
train_encodings = tokenizer(train['Text'].tolist(), padding=True, truncation=True, 
                            max_length=512, return_tensors="pt")
val_encodings = tokenizer(val['Text'].tolist(), padding=True, truncation=True, 
                          max_length=512, return_tensors="pt")

train_labels = torch.tensor(train['Label'].tolist())
val_labels = torch.tensor(val['Label'].tolist())

# Create TensorDatasets for training and validation
train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)
val_dataset = TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], val_labels)

# Create DataLoaders for batching
train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=bs)
val_loader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=bs)

# Initialize the BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-large-multilingual-cased', num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set up the optimizer
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

# Track training time
start_time = time.time()
epoch_stats = []  # To store training statistics for each epoch

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        
        optimizer.zero_grad()
        outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_train_loss = total_loss / len(train_loader)

    # Validation loop
    model.eval()
    total_val_loss = 0
    predictions, confidences, indexes = [], [], []
    
    for i, batch in enumerate(val_loader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        indexes_batch = val['EstText Index'][i*bs:(i+1)*bs].tolist()

        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss
            total_val_loss += loss.item()
            
            logits = outputs.logits.detach()
            probs = F.softmax(logits, dim=1)
            batch_confidences = probs.max(dim=1)[0]  
            batch_predictions = logits.argmax(dim=1)

            predictions.extend(batch_predictions.tolist())
            confidences.extend(batch_confidences.tolist())
            indexes.extend(indexes_batch)

    avg_val_loss = total_val_loss / len(val_loader)
    val_accuracy = accuracy_score(val_labels, predictions)
    
    epoch_stats.append({
        'Epoch': epoch + 1,
        'Training Loss': avg_train_loss,
        'Validation Loss': avg_val_loss,
        'Validation Accuracy': val_accuracy
    })

    print(f'Epoch {epoch + 1}: Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}, Validation Accuracy: {val_accuracy}')

# Save training statistics and model outputs
training_time = time.time() - start_time
timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M')

# Save epoch stats to a Parquet file
stats_df = pd.DataFrame(epoch_stats)
stats_df.to_parquet(os.path.join(data_dir, f'{mn}-epoch-stats-{timestamp}.parquet'), index=False)

# Save validation predictions to a Parquet file
results_df = pd.DataFrame({
    'EstText Index': indexes,
    'Label': predictions,
    'Confidence': confidences
})
results_df.to_parquet(os.path.join(data_dir, f'{mn}-val-predictions-{timestamp}.parquet'), index=False)

# Save the model state dict
torch.save(model.state_dict(), f'{mn}-model-state-dict.pth')

print(f'Training of {mn} completed in {training_time:.2f} seconds')