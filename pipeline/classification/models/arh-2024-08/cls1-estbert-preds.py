# EstBERT predictions

import time
import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from textclean import cleancolumn

# Parameters
mn = 'EstBERT'
model_name = f'tartuNLP/{mn}'
data_dir = '../data'  
bs = 64  

# Load and clean the test data
test = pd.read_parquet(os.path.join(data_dir, 'cls1-sim-output-est.parquet')) 
test = cleancolumn(test, 'Text')

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# Initialize the model and load the pre-trained weights
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.load_state_dict(torch.load(f'{mn}-model-state-dict.pth', map_location=device))  
model.to(device)  

# Tokenize the text data and create a dataset for input to the model
test_encodings = tokenizer(test['Text'].tolist(), padding=True, truncation=True, max_length=512, return_tensors="pt")
test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'])

test_loader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=bs)

start_time = time.time()

model.eval()
predictions, confidences, indexes = [], [], []

# Loop over batches of data
for i, batch in enumerate(test_loader):
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask = batch
    
    # Generate indexes for the current batch
    indexes_batch = test['EstText Index'][i*bs:(i+1)*bs].tolist() if 'EstText Index' in test else list(range(i*bs, (i+1)*bs))
    
    with torch.no_grad():
        outputs = model(b_input_ids, attention_mask=b_input_mask)
        logits = outputs.logits
        
        probs = F.softmax(logits, dim=1)
        
        # Get the maximum probability and its corresponding class prediction
        batch_confidences = probs.max(dim=1)[0]
        batch_predictions = logits.argmax(dim=1)
        
        predictions.extend(batch_predictions.tolist())
        confidences.extend(batch_confidences.tolist())
        indexes.extend(indexes_batch)

pred_time = time.time() - start_time

results_df = pd.DataFrame({'EstText Index': indexes, 'Prediction': predictions, 'Confidence': confidences})
results_df.to_parquet(os.path.join(data_dir, f'{mn}-test-predictions.parquet'), index=False)

print(f'Predictions and confidences from {mn} saved to file.')
print(f'Predicting of {mn} completed in {pred_time:.2f} seconds')