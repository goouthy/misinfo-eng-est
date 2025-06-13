# MT5 Base Model 

import time
import os
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import MT5Tokenizer, MT5Model
from sklearn.metrics import accuracy_score

from textclean import cleancolumn
from trainsplit import splitandbalance

# Parameters
model_name = 'google/mt5-base'  
data_dir = '../data'
batch_size = 64
epochs = 12
learning_rate = 1e-5
num_labels = 2  

# Load and preprocess data
input_data = pd.read_parquet(os.path.join(data_dir, 'goldstandard-dataset.parquet'))
input_data = cleancolumn(input_data, 'Text')

# Split and balance dataset
train_df, val_df = splitandbalance(input_data)

# Initialize tokenizer
tokenizer = MT5Tokenizer.from_pretrained(model_name)

# Tokenize text data
train_encodings = tokenizer(train_df['Text'].tolist(), padding=True, truncation=True, max_length=512, return_tensors="pt")
val_encodings = tokenizer(val_df['Text'].tolist(), padding=True, truncation=True, max_length=512, return_tensors="pt")

# Convert labels to tensors
train_labels = torch.tensor(train_df['Label'].tolist())
val_labels = torch.tensor(val_df['Label'].tolist())

# Create TensorDatasets
train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)
val_dataset = TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], val_labels)

# Create DataLoaders
train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
val_loader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)

# Define model
class MT5ForSequenceClassification(nn.Module):
    def __init__(self, model_name, num_labels):
        super(MT5ForSequenceClassification, self).__init__()
        self.mt5 = MT5Model.from_pretrained(model_name)
        self.classification_head = nn.Linear(self.mt5.config.d_model, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.mt5(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        pooled_output = hidden_states.mean(dim=1)  # Mean pooling

        logits = self.classification_head(pooled_output)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.classification_head.out_features), labels.view(-1))

        return (loss, logits) if loss is not None else logits

# Initialize model
model = MT5ForSequenceClassification(model_name, num_labels=num_labels)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set up optimizer
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)

# Track training time
start_time = time.time()
epoch_stats = []

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        
        optimizer.zero_grad()
        outputs = model(input_ids=b_input_ids, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs[0]
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # Validation phase
    model.eval()
    total_val_loss = 0
    predictions = []
    indexes = []

    for i, batch in enumerate(val_loader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        indexes_batch = val_df['EstText Index'][i*batch_size:(i+1)*batch_size].tolist()

        with torch.no_grad():
            outputs = model(input_ids=b_input_ids, attention_mask=b_input_mask, labels=b_input_ids)
            loss = outputs[0]
            total_val_loss += loss.item()
            
            logits = outputs[1]
            batch_predictions = torch.argmax(logits, dim=1)

            predictions.extend(batch_predictions.cpu().tolist())
            indexes.extend(indexes_batch)

    avg_val_loss = total_val_loss / len(val_loader)

    val_labels_list = val_labels.cpu().tolist()
    val_accuracy = accuracy_score(val_labels_list, predictions)

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
stats_df.to_parquet(os.path.join(data_dir, f'{model_name}-epoch-stats.parquet'), index=False)

results_df = pd.DataFrame({
    'EstText Index': indexes,
    'Prediction': predictions
})
results_df.to_parquet(os.path.join(data_dir, f'{model_name}-val-predictions.parquet'), index=False)

torch.save(model.state_dict(), f'{model_name}-model-state-dict.pth')

print(f'Training of {model_name} completed in {training_time:.2f} seconds')