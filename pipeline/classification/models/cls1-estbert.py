# EstBERT Model

import time
import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score

from textclean import cleancolumn
from trainsplit import splitandbalance

# Parameters
mn = 'EstBERT'
model_name = f'tartuNLP/{mn}'  
data_dir = '../data'  
bs = 64  
epochs = 12
lr = 1e-5  

# Load and process data
input = pd.read_parquet(os.path.join(data_dir, 'goldstandard-dataset.parquet')) 
input = cleancolumn(input, 'Text')  

train, val = splitandbalance(input)

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize the text data and create datasets for input to the model
train_encodings = tokenizer(train['Text'].tolist(), padding=True, truncation=True, 
                            max_length=512, return_tensors="pt")
val_encodings = tokenizer(val['Text'].tolist(), padding=True, truncation=True, 
                          max_length=512, return_tensors="pt")

train_labels = torch.tensor(train['Label'].tolist())
val_labels = torch.tensor(val['Label'].tolist())

train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)
val_dataset = TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], val_labels)

train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=bs)
val_loader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=bs)

# Initialize model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
model.to(device)

# Optimizer - AdamW
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

start_time = time.time()
epoch_stats = []

# Train
for epoch in range(epochs):
    model.train()  
    total_loss = 0 

    for batch in train_loader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        
        optimizer.zero_grad()
        # Forward pass
        outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
        # Extract the loss from the model's outputs
        loss = outputs.loss          
        # Perform backpropagation
        loss.backward()        
        # Update model parameters
        optimizer.step()        
        # Accumulate the total loss for this epoch
        total_loss += loss.item()
    
    # Calculate the average training loss for this epoch
    avg_train_loss = total_loss / len(train_loader)

    # Validate
    total_val_loss = 0  
    predictions, confidences, indexes = [], [], []  
    model.eval() 
    
    for i, batch in val_loader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        
        # Generate the corresponding indexes for this batch
        indexes_batch = val['EstText Index'][i*bs:(i+1)*bs].tolist()

        # Disable gradient calculation 
        with torch.no_grad():
            # Forward pass 
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss  # Extract the loss for this batch
            total_val_loss += loss.item()  # Accumulate the validation loss
            
            # Extract logits and compute probabilities
            logits = outputs.logits.detach()  # Detach to avoid backprop
            probs = F.softmax(logits, dim=1)  # Convert logits to probabilities
            
            # Determine the most confident prediction and its corresponding probability
            batch_confidences = probs.max(dim=1)[0]
            batch_predictions = logits.argmax(dim=1)
            
            predictions.extend(batch_predictions.tolist())
            confidences.extend(batch_confidences.tolist())
            indexes.extend(indexes_batch)

    # Calculate the average validation loss for this epoch
    avg_val_loss = total_val_loss / len(val_loader)
    
    # Compute the accuracy on the validation set
    val_accuracy = accuracy_score(val_labels.cpu().numpy(), predictions)
    
    epoch_stats.append({
        'Epoch': epoch + 1,
        'Training Loss': avg_train_loss,
        'Validation Loss': avg_val_loss,
        'Validation Accuracy': val_accuracy
    })

    print(f'Epoch {epoch + 1}: Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}, Validation Accuracy: {val_accuracy}')

training_time = time.time() - start_time

stats_df = pd.DataFrame(epoch_stats)
stats_df.to_parquet(os.path.join(data_dir, f'{mn}-epoch-stats.parquet'), index=False)

results_df = pd.DataFrame({
    'EstText Index': indexes,
    'Prediction': predictions,
    'Confidence': confidences
})
results_df.to_parquet(os.path.join(data_dir, f'{mn}-val-predictions.parquet'), index=False)

# Save model for predicting
torch.save(model.state_dict(), f'{mn}-model-state-dict.pth')

print(f'Training of {mn} completed in {training_time:.2f} seconds')