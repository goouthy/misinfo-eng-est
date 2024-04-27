# XLM-RoBERTa predictions

from datetime import datetime
import time

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification

from textclean import cleancolumn

mn = 'XLM-RoBERTa'
bs = 64

test = pd.read_csv('cls2-sim-output-est.csv')

tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-large', num_labels=2)
model.load_state_dict(torch.load(f'{mn}_model_state_dict.pth', map_location=device))
model.to(device)

test_encodings = tokenizer(test['Text'].tolist(), padding=True, truncation=True, max_length=512, return_tensors="pt")
test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'])
test_loader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=bs)

model.eval()
predictions, confidences, indexes = [], [], []

start_time = time.time()

for i, batch in enumerate(test_loader):
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask = batch
    indexes_batch = test['EstText Index'][i*bs:(i+1)*bs].tolist() if 'EstText Index' in test else list(range(i*bs, (i+1)*bs))
    
    with torch.no_grad():
        outputs = model(b_input_ids, attention_mask=b_input_mask)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1) 
        batch_confidences = probs.max(dim=1)[0]
        batch_predictions = logits.argmax(dim=1)  
        
        predictions.extend(batch_predictions.tolist())
        confidences.extend(batch_confidences.tolist())
        indexes.extend(indexes_batch)

timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M')
results_df = pd.DataFrame({'EstText Index': indexes, 'Prediction': predictions, 'Confidence': confidences})
results_df.to_csv(f'{mn}_test_predictions_{timestamp}.csv', index=False)

print(f'Predictions and confidences from {mn} saved to file at {timestamp}.')
print(f'Prediction took {time.time() - start_time:.2f} seconds.')
