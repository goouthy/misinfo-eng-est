import pandas as pd
import string
import re

def cleansingletext(input_text):
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

def cleancolumn(df, column_name):
    ''' Cleaning only the text column in input dataframe '''
    df_cleaned = df.copy()
    df_cleaned[column_name] = df_cleaned[column_name].apply(cleansingletext).astype(str)
    df_cleaned = df_cleaned.dropna(subset=[column_name]).drop_duplicates(subset=[column_name])
    return df_cleaned
