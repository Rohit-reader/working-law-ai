import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import joblib
import spacy
from pathlib import Path

def load_and_preprocess_data(data_path):
    """Load and preprocess the legal dataset."""
    # TODO: Replace with actual data loading logic
    # This is a placeholder for demonstration
    data = pd.read_csv(data_path)
    return data

def train_model(X_train, y_train):
    """Train a text classification model."""
    # Convert text to TF-IDF features
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    
    # Train a simple LinearSVC model
    model = LinearSVC(random_state=42, max_iter=10000)
    model.fit(X_train_vec, y_train)
    
    return model, vectorizer

def evaluate_model(model, vectorizer, X_test, y_test):
    """Evaluate the trained model."""
    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)
    
    print("\nModel Evaluation:")
    print(classification_report(y_test, y_pred))

def save_model(model, vectorizer, model_dir='model'):
    """Save the trained model and vectorizer."""
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    model_path = os.path.join(model_dir, 'legal_classifier.joblib')
    vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.joblib')
    
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"\nModel saved to {model_path}")
    print(f"Vectorizer saved to {vectorizer_path}")

def main():
    # Configuration
    DATA_PATH = os.path.join('data', 'raw', 'legal_cases.csv')
    MODEL_DIR = 'saved_models'
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    data = load_and_preprocess_data(DATA_PATH)
    
    # Split data (assuming 'text' and 'label' columns exist)
    X_train, X_test, y_train, y_test = train_test_split(
        data['text'], data['label'], test_size=0.2, random_state=42
    )
    
    # Train model
    print("Training model...")
    model, vectorizer = train_model(X_train, y_train)
    
    # Evaluate model
    evaluate_model(model, vectorizer, X_test, y_test)
    
    # Save model
    save_model(model, vectorizer, MODEL_DIR)

if __name__ == "__main__":
    main()
