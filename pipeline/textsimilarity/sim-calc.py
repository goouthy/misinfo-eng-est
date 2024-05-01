import pandas as pd
import time
import re
import string
from itertools import product
from sentence_transformers import SentenceTransformer, models, util

# Data
estonian_news = pd.read_csv('../est-dataset/all_est_news.csv')
english_news = pd.read_csv('../eng-dataset/all_eng_news_texts.csv')

estonian_news = estonian_news.reset_index().rename(columns={'index': 'EstText Index'})
english_news = english_news.reset_index().rename(columns={'index': 'EngText Index'})

def clean_single_text(input_text):
    ''' Cleaning texts by sentences: removing hyperlinks, punctuation, too short sentences '''
    if pd.isna(input_text):
        return None
    
    input_text = input_text.lower()
    input_text = re.sub(r'\bhttp\S+|www\S+\b', '', input_text)
    lines = input_text.split('\n')
    filtered_lines = [line.strip() for line in lines if len(line.split()) >= 5]
    cleaned_text = ' '.join(filtered_lines)
    translation_table = str.maketrans('', '', string.punctuation)
    cleaned_sentence = cleaned_text.translate(translation_table)
    return cleaned_sentence

def clean_column(df, column_name):
    ''' Cleaning only the text column in input dataframe '''
    df_cleaned = df.copy()
    df_cleaned[column_name] = df_cleaned[column_name].apply(clean_single_text)
    df_cleaned = df_cleaned.dropna(subset=[column_name]).drop_duplicates(subset=[column_name])
    return df_cleaned

# Clean data
est_news_cleaned = clean_column(estonian_news, 'Text').reset_index(drop=True)
eng_news_cleaned = clean_column(english_news, 'text').reset_index(drop=True)

# Subsets of article text and index
esttexts = est_news_cleaned[['EstText Index', 'Text']]
engtexts = eng_news_cleaned[['EngText Index', 'text']]

print(f'EstText count: {len(esttexts)}')
print(f'EngText count: {len(engtexts)}')

def get_embeddings(text_chunks, model):
    ''' Generating embeddings in batch size 1000 '''
    chunk_embeddings = model.encode(text_chunks, convert_to_tensor=True, batch_size=1000, show_progress_bar=True)
    return chunk_embeddings

# SBERT configuration
embedding_model = models.Transformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2', 
                                          max_seq_length=512, do_lower_case=True)
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2', modules=[embedding_model])

# Start of processing
start_time = time.time()

# Generating sentence embeddings for engtexts and esttexts
embeddings_eng = get_embeddings(engtexts['text'], model)
embeddings_est = get_embeddings(esttexts['Text'], model)

print('Total embeddings generation time:', (time.time() - start_time) / 60, 'minutes')

# Start of similarity calculation
similarity_start_time = time.time()

cosine_scores = util.cos_sim(embeddings_eng, embeddings_est)

indices_i, indices_j, cossin_scores = [], [], []
for i, j in product(range(len(engtexts)), range(len(esttexts))):
    indices_i.append(engtexts.iloc[i]['EngText Index'])
    indices_j.append(esttexts.iloc[j]['EstText Index'])
    cossin_scores.append(cosine_scores[i][j].item())

similarity_results = pd.DataFrame({
    'EngText Index': indices_i,
    'EstText Index': indices_j,
    'Cosine Similarity Score': cossin_scores
})

similarity_results.to_parquet('similarity_results.parquet', index=False, compression='snappy', engine='pyarrow')

print('Similarity scores saved to similarity_results.parquet')

total_similarity_calc_time = time.time() - similarity_start_time
print('Total similarity calculation time: ', total_similarity_calc_time / 60, 'minutes')
print('Total execution time: ', (time.time() - start_time) / 60, 'minutes')

# Thresholding
threshold = 0.5

scores_combined = pd.read_parquet('similarity_results.parquet')

print(f'Shape of all possible score dataset: {scores_combined.shape}')

over_threshold = scores_combined[scores_combined['Cosine Similarity Score'] > threshold]
over_threshold.to_parquet(f'over_threshold.parquet', index=False, compression='snappy', engine='pyarrow')

print(f'Shape of all scores above threshold {threshold} dataset: {over_threshold.shape}')