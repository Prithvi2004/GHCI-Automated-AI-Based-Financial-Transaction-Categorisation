# src/preprocess.py
import re
import string

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove special characters except spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def preprocess_dataframe(df, text_col='description'):
    df = df.copy()
    df[text_col] = df[text_col].apply(clean_text)
    return df