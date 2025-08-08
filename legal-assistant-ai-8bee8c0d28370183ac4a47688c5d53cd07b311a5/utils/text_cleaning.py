import re
import string
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    # If the model is not found, download it
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load('en_core_web_sm')

class TextCleaner:
    def __init__(self, remove_stopwords=True, lemmatize=True):
        """Initialize the text cleaner with processing options.
        
        Args:
            remove_stopwords: Whether to remove stopwords
            lemmatize: Whether to lemmatize words
        """
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Add legal-specific stopwords
        legal_stopwords = {
            'herein', 'hereby', 'hereto', 'hereof', 'hereinbefore', 'hereinafter',
            'thereof', 'thereby', 'therein', 'thereto', 'whereas', 'whereby',
            'witnesseth', 'pursuant', 'notwithstanding', 'aforesaid'
        }
        self.stop_words.update(legal_stopwords)
    
    def clean_text(self, text):
        """Clean and preprocess the input text."""
        if not isinstance(text, str) or not text.strip():
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Lemmatization
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Join tokens back into a single string
        return ' '.join(tokens)
    
    def extract_entities(self, text):
        """Extract named entities from the text using spaCy."""
        doc = nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
            
        return entities
    
    def extract_key_phrases(self, text, top_n=5):
        """Extract key phrases from the text."""
        doc = nlp(text)
        
        # Extract noun chunks as potential key phrases
        key_phrases = [chunk.text.lower() for chunk in doc.noun_chunks]
        
        # Count phrase frequencies
        phrase_freq = {}
        for phrase in key_phrases:
            if phrase not in phrase_freq:
                phrase_freq[phrase] = 0
            phrase_freq[phrase] += 1
        
        # Sort by frequency and get top N
        sorted_phrases = sorted(phrase_freq.items(), key=lambda x: x[1], reverse=True)
        return [phrase for phrase, _ in sorted_phrases[:top_n]]

# Example usage
if __name__ == "__main__":
    cleaner = TextCleaner()
    
    sample_text = """
    The plaintiff, John Doe, filed a lawsuit against XYZ Corporation on January 15, 2023,
    alleging breach of contract. The defendant has 30 days to respond to the complaint.
    """
    
    print("Original Text:")
    print(sample_text)
    print("\nCleaned Text:")
    print(cleaner.clean_text(sample_text))
    print("\nEntities:")
    print(cleaner.extract_entities(sample_text))
    print("\nKey Phrases:")
    print(cleaner.extract_key_phrases(sample_text))
