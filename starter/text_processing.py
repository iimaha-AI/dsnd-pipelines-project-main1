import re
import pandas as pd
import spacy
from sklearn.base import BaseEstimator, TransformerMixin

class TextProcessor(BaseEstimator, TransformerMixin):
    """Custom text processor using spaCy for NLP tasks"""
    
    def __init__(self):
        self.nlp = None
    
    def fit(self, X, y=None):
        # Load the spaCy model with disabled components for efficiency
        try:
            self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        except OSError as exc:
            raise OSError("Install the required language model first: python -m spacy download en_core_web_sm") from exc
        return self
    
    def process_text(self, text):
        """Process a single text document"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
        
        # Process with spaCy
        doc = self.nlp(text)
        
        # Remove stop words, punctuation, and lemmatize
        processed_tokens = [
            token.lemma_ for token in doc 
            if not token.is_stop and not token.is_punct and len(token) > 2
        ]
        
        return " ".join(processed_tokens)
    
    def transform(self, X, y=None):
        """Process a collection of text documents"""
        return [self.process_text(text) for text in X]
