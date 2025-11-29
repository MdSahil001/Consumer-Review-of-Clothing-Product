import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

class ReviewPreprocessor:
    """
    Handles the cleaning and tokenization of raw text reviews.
    """
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """
        Input: Raw string
        Output: List of cleaned tokens
        """
        # Handle NaN or non-string inputs
        if not isinstance(text, str):
            return
            
        # 1. Lowercase normalization
        text = text.lower()
        
        # 2. Remove non-alphabetic characters (keep spaces)
        # We strip punctuation as it rarely helps in embedding averaging
        text = re.sub(r'[^a-z\s]', '', text)
        
        # 3. Tokenization
        tokens = text.split()
        
        # 4. Stopword removal and Lemmatization
        clean_tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return clean_tokens