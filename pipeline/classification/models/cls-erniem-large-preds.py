# ERNIE-M Predictions on Test Set

import time
import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from textclean import cleancolumn

# Parameters
step = 'second'

mn = 'ernie-m-large'
model_name = 'susnato/ernie-m-large_pytorch' 
data_dir = '../data'
bs = 64 

if step == 'first':
    # First step classification parameters:
    input = 'cls1-output.parquet'
    model_version = f'{mn}-model-state-dict.pth'
    output = f'cls1-{mn}-test-predictions.parquet'

elif step == 'second':
    # Second step classification parameters:
    input = 'cls2-output.parquet'
    model_version = f'cls2-{mn}-model-state-dict-2.pth'
    output = f'cls2-{mn}-test-predictions.parquet'
    

# Load and process test data
test = pd.read_parquet(os.path.join(data_dir, input))
test = cleancolumn(test, 'Text')

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model and load the pre-trained weights
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.load_state_dict(torch.load(model_version, map_location=device))
model.to(device)

# Tokenize the test data and create a dataset for input to the model
test_encodings = tokenizer(test['Text'].tolist(), padding=True, truncation=True, max_length=512, return_tensors="pt", return_attention_mask=True)
test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'])

test_loader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=bs)

start_time = time.time()

# Evaluation
model.eval()  
predictions, confidences, indexes = [], [], []

# Loop over batches of test data
for i, batch in enumerate(test_loader):
    batch = tuple(t.to(device) for t in batch)  
    b_input_ids, b_input_mask = batch
    
    # Generate corresponding indexes for this batch (based on the presence of 'EstText Index' in the test DataFrame)
    indexes_batch = test['EstText Index'][i*bs:(i+1)*bs].tolist() if 'EstText Index' in test else list(range(i*bs, (i+1)*bs))
    
    with torch.no_grad():  # Disable gradient computation for inference
        outputs = model(b_input_ids, attention_mask=b_input_mask)  # Forward pass
        logits = outputs.logits  # Get raw predictions (logits)
        
        probs = F.softmax(logits, dim=1)  # Convert logits to probabilities
        
        # Get the maximum probability and its corresponding class prediction
        batch_confidences = probs.max(dim=1)[0]
        batch_predictions = logits.argmax(dim=1)
        
        predictions.extend(batch_predictions.tolist())
        confidences.extend(batch_confidences.tolist())
        indexes.extend(indexes_batch)

pred_time = time.time() - start_time

results_df = pd.DataFrame({'EstText Index': indexes, 'Prediction': predictions, 'Confidence': confidences})
results_df.to_parquet(os.path.join(data_dir, output), index=False)

print(f'Predictions and confidences from {mn} saved to file.')
print(f'Predicting of {mn} completed in {pred_time:.2f} seconds')