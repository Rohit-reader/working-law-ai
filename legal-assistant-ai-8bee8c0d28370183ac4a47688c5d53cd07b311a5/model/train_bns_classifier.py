import os
import pandas as pd
import joblib
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pathlib import Path

class BNSClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.classifier = LinearSVC(random_state=42, max_iter=10000)
        self.label_encoder = {}
        self.reverse_label_encoder = {}
        self.nlp = spacy.load('en_core_web_sm')
    
    def preprocess_text(self, text):
        """Preprocess the input text using spaCy"""
        doc = self.nlp(text.lower())
        # Lemmatize and remove stopwords and punctuation
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        return ' '.join(tokens)
    
    def fit(self, X, y):
        """Train the classifier"""
        # Create label encoding
        self.label_encoder = {label: idx for idx, label in enumerate(sorted(set(y)))}
        self.reverse_label_encoder = {idx: label for label, idx in self.label_encoder.items()}
        
        # Transform labels to numerical values
        y_encoded = [self.label_encoder[label] for label in y]
        
        # Vectorize the text data
        X_vec = self.vectorizer.fit_transform(X)
        
        # Train the classifier
        self.classifier.fit(X_vec, y_encoded)
        
        return self
    
    def predict(self, X):
        """Predict BNS sections for input texts"""
        X_vec = self.vectorizer.transform(X)
        y_pred = self.classifier.predict(X_vec)
        return [self.reverse_label_encoder[pred] for pred in y_pred]
    
    def save(self, model_dir='saved_models'):
        """Save the model and vectorizer"""
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        model_path = os.path.join(model_dir, 'bns_classifier.joblib')
        vectorizer_path = os.path.join(model_dir, 'bns_vectorizer.joblib')
        
        joblib.dump({
            'classifier': self.classifier,
            'label_encoder': self.label_encoder,
            'reverse_label_encoder': self.reverse_label_encoder
        }, model_path)
        
        joblib.dump(self.vectorizer, vectorizer_path)
        
        return model_path, vectorizer_path
    
    @classmethod
    def load(cls, model_dir='saved_models'):
        """Load a trained model"""
        model_path = os.path.join(model_dir, 'bns_classifier.joblib')
        vectorizer_path = os.path.join(model_dir, 'bns_vectorizer.joblib')
        
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            raise FileNotFoundError("Model or vectorizer not found. Please train the model first.")
        
        model_data = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        
        classifier = cls()
        classifier.classifier = model_data['classifier']
        classifier.label_encoder = model_data['label_encoder']
        classifier.reverse_label_encoder = model_data['reverse_label_encoder']
        classifier.vectorizer = vectorizer
        
        return classifier

def load_bns_dataset(data_path):
    """Load and preprocess the BNS dataset"""
    df = pd.read_csv(data_path)
    return df

def train_bns_classifier():
    # Configuration
    DATA_PATH = os.path.join('data', 'raw', 'bns_dataset.csv')
    MODEL_DIR = 'saved_models'
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_bns_dataset(DATA_PATH)
    
    # Initialize and train the classifier
    print("Training BNS classifier...")
    classifier = BNSClassifier()
    
    # Preprocess text
    X = [classifier.preprocess_text(text) for text in df['incident_text']]
    y = df['section'].astype(str) + " - " + df['section_title']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train the model
    classifier.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = classifier.predict(X_test)
    print("\nModel Evaluation:")
    print(classification_report(y_test, y_pred))
    
    # Save the model
    model_path, vectorizer_path = classifier.save(MODEL_DIR)
    print(f"\nModel saved to {model_path}")
    print(f"Vectorizer saved to {vectorizer_path}")
    
    # Create a mapping file for section details
    section_details = df[['section', 'section_title', 'description', 'punishment']].drop_duplicates()
    section_details.to_csv(os.path.join(MODEL_DIR, 'bns_section_details.csv'), index=False)
    print(f"Section details saved to {os.path.join(MODEL_DIR, 'bns_section_details.csv')}")
    
    return classifier

if __name__ == "__main__":
    train_bns_classifier()
