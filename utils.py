
import joblib

import re
import string
from nltk.corpus import stopwords



def load_model(model_path):
    """
    Load a joblib model

    Args:
    - model_path (str): path to the model

    Returns:
    - model: loaded model
    """
    model = joblib.load(model_path)
    return model



# Set of English stopwords
stop_words = set(stopwords.words('english'))

def preprocess_text(text:str):
    # Step 1: Lowercase
    text = text.lower()

    # Step 2: Strip extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())

    # Step 3: Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Step 4: Remove stopwords
    text = ' '.join(word for word in text.split() if word not in stop_words)

    # Step 5: Remove noise (URLs, emails, hashtags, mentions, numbers, non-printables)
    text = re.sub(r'http\S+|www\.\S+', '', text)       # URLs
    text = re.sub(r'\S+@\S+\.\S+', '', text)           # Emails
    text = re.sub(r'#[A-Za-z0-9_]+', '', text)         # Hashtags
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)         # Mentions
    text = re.sub(r'\d+', '', text)                    # Numbers
    text = ''.join(ch for ch in text if ch.isprintable())  # Non-printables

    return text

