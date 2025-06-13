# REMBERT Base Model

import os
import time
import pandas as pd
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import ReformerTokenizer, ReformerForSequenceClassification
from sklearn.metrics import accuracy_score

from textclean import cleancolumn
from trainsplit import splitandbalance

# Parameters
mn = 'rembert-base' 
model = 'google/rembert-base' 
data_dir = '../data'
bs = 64
epochs = 12
lr = 1e-5

# Load and preprocess data
input = pd.read_parquet(os.path.join(data_dir, 'goldstandard-dataset.parquet'))
input = cleancolumn(input, 'Text')

# Split and balance dataset
train, val = splitandbalance(input)

# Initialize tokenizer
tokenizer = ReformerTokenizer.from_pretrained(model)

# Tokenize text data
train_encodings = tokenizer(train['Text'].tolist(), padding=True, truncation=True, 
                            max_length=512, return_tensors="pt")
val_encodings = tokenizer(val['Text'].tolist(), padding=True, truncation=True, 
                          max_length=512, return_tensors="pt")

# Convert labels to tensors
train_labels = torch.tensor(train['Label'].tolist())
val_labels = torch.tensor(val['Label'].tolist())

# Create TensorDatasets
train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)
val_dataset = TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], val_labels)

# Create DataLoaders
train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=bs)
val_loader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=bs)

# Initialize model
model = ReformerForSequenceClassification.from_pretrained(model, num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set up optimizer
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

# Track training time
start_time = time.time()
epoch_stats = []

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    # Train
    for batch in train_loader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        
        optimizer.zero_grad()
        outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_train_loss = total_loss / len(train_loader)

    # Validate
    model.eval()
    total_val_loss = 0
    predictions, confidences, indexes = [], [], []
    
    for i, batch in enumerate(val_loader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        indexes_batch = val['EstText Index'][i*bs:(i+1)*bs].tolist()

        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_input_ids)
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

# Save results
training_time = time.time() - start_time

stats_df = pd.DataFrame(epoch_stats)
stats_df.to_parquet(os.path.join(data_dir, f'{mn}-epoch-stats.parquet'), index=False)

results_df = pd.DataFrame({
    'EstText Index': indexes,
    'Prediction': predictions,
    'Confidence': confidences
})
results_df.to_parquet(os.path.join(data_dir, f'{mn}-val-predictions.parquet'), index=False)

torch.save(model.state_dict(), f'{mn}-model-state-dict.pth')

print(f'Training of {mn} completed in {training_time:.2f} seconds')