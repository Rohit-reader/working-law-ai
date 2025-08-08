import os
import pandas as pd
import joblib
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path
import re
import random
from tqdm import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

class EnhancedBNSClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        self.classifier = LinearSVC(
            class_weight='balanced',
            random_state=42,
            max_iter=5000,
            C=0.5
        )
        self.nlp = spacy.load('en_core_web_sm')
        self.section_details = {}
        
    def preprocess_text(self, text):
        """Preprocess the input text using spaCy"""
        if not isinstance(text, str) or not text.strip():
            return ""
            
        # Basic cleaning
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
        
        # Use spaCy for advanced preprocessing
        doc = self.nlp(text)
        
        # Lemmatize and filter tokens
        tokens = []
        for token in doc:
            if not token.is_stop and not token.is_punct and not token.is_space:
                lemma = token.lemma_.lower().strip()
                if lemma and len(lemma) > 2:  # Remove very short tokens
                    tokens.append(lemma)
        
        return ' '.join(tokens)
    
    def load_data(self, data_path):
        """Load and preprocess the BNS dataset with data augmentation"""
        # Load the dataset
        df = pd.read_csv(data_path)
        df = df.dropna(subset=['description', 'example_incidents', 'punishment'])
        
        # Store section details for later use
        self.section_details = df.set_index('section').to_dict('index')
        
        texts = []
        labels = []
        
        # Process each section
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing sections"):
            section_id = row['section']
            section_title = row['title']
            label = f"{section_id} - {section_title}"
            
            # Get examples and clean them
            section_examples = []
            if pd.notna(row['example_incidents']):
                section_examples = [ex.strip() for ex in str(row['example_incidents']).split(';') if ex.strip()]
            
            # Add the main description as an example
            if pd.notna(row['description']):
                section_examples.append(row['description'])
            
            # Add punishment information as context
            if pd.notna(row['punishment']):
                section_examples.append(f"Punishment: {row['punishment']}")
            
            # Augment the data by creating variations of examples
            augmented_examples = []
            for example in section_examples:
                # Add the original example
                augmented_examples.append(example)
                
                # Create variations with different phrasing
                if len(augmented_examples) < 10:  # Limit augmentation to prevent class imbalance
                    words = example.split()
                    if len(words) > 5:
                        # Variation 1: Remove middle part
                        if len(words) > 10:
                            augmented = ' '.join(words[:3] + words[-3:])
                            augmented_examples.append(augmented)
                        
                        # Variation 2: Shuffle words (if makes sense)
                        if len(words) < 15:
                            shuffled = words.copy()
                            random.shuffle(shuffled)
                            if ' '.join(shuffled) != example:  # Only add if different
                                augmented_examples.append(' '.join(shuffled))
            
            # Add to final dataset
            for example in augmented_examples:
                if example:  # Skip empty examples
                    processed_text = self.preprocess_text(example)
                    if processed_text:  # Only add if we have content after preprocessing
                        texts.append(processed_text)
                        labels.append(label)
        
        return texts, labels, df
    
    def train(self, X, y):
        """Train the classifier with balanced class weights"""
        # Vectorize the text data
        X_vec = self.vectorizer.fit_transform(X)
        
        # Calculate class weights to handle imbalance
        classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weight_dict = dict(zip(classes, class_weights))
        
        # Update classifier with balanced class weights
        self.classifier.class_weight = class_weight_dict
        
        # Train the classifier with cross-validation
        print("\nTraining classifier with cross-validation...")
        cv_scores = cross_val_score(
            self.classifier, X_vec, y, 
            cv=min(5, len(np.unique(y))),  # Use min of 5 or number of classes
            scoring='f1_weighted',
            n_jobs=-1
        )
        
        print(f"\nCross-validation F1 scores: {cv_scores}")
        print(f"Mean CV F1 score: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
        
        # Final training on full dataset
        self.classifier.fit(X_vec, y)
        
        return self
    
    def evaluate(self, X_test, y_test):
        """Evaluate the classifier"""
        X_test_vec = self.vectorizer.transform(X_test)
        y_pred = self.classifier.predict(X_test_vec)
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
    
    def save(self, model_dir='saved_models'):
        """Save the model, vectorizer, and section details"""
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        
        # Save the model components
        model_path = os.path.join(model_dir, 'enhanced_bns_classifier.joblib')
        vectorizer_path = os.path.join(model_dir, 'enhanced_bns_vectorizer.joblib')
        details_path = os.path.join(model_dir, 'bns_section_details_enhanced.csv')
        
        joblib.dump(self.classifier, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        
        # Save section details as a DataFrame
        details_df = pd.DataFrame.from_dict(self.section_details, orient='index')
        details_df.to_csv(details_path)
        
        print(f"\nModel saved to {model_path}")
        print(f"Vectorizer saved to {vectorizer_path}")
        print(f"Section details saved to {details_path}")
        
        return model_path, vectorizer_path, details_path

    @classmethod
    def load(cls, model_dir='saved_models'):
        """Load a trained model"""
        model_path = os.path.join(model_dir, 'enhanced_bns_classifier.joblib')
        vectorizer_path = os.path.join(model_dir, 'enhanced_bns_vectorizer.joblib')
        details_path = os.path.join(model_dir, 'bns_section_details_enhanced.csv')
        
        if not all(os.path.exists(p) for p in [model_path, vectorizer_path, details_path]):
            raise FileNotFoundError("One or more model files not found. Please train the model first.")
        
        # Create a new instance
        classifier = cls()
        
        # Load the components
        classifier.classifier = joblib.load(model_path)
        classifier.vectorizer = joblib.load(vectorizer_path)
        
        # Load section details
        details_df = pd.read_csv(details_path, index_col=0)
        classifier.section_details = details_df.to_dict('index')
        
        return classifier

def main():
    # Configuration
    DATA_PATH = os.path.join('data', 'raw', 'bns_sections_extended.csv')
    MODEL_DIR = 'saved_models'
    
    # Create output directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Initialize the classifier
    print("="*80)
    print("Initializing Enhanced BNS Classifier")
    print("="*80)
    classifier = EnhancedBNSClassifier()
    
    try:
        # Load and preprocess data
        print("\n" + "-"*40)
        print("Loading and preprocessing data...")
        print("-"*40)
        X, y, section_details = classifier.load_data(DATA_PATH)
        
        # Print class distribution
        print("\nClass distribution in dataset:")
        class_counts = pd.Series(y).value_counts().sort_index()
        print(class_counts)
        
        # Split data into train and test sets
        # Use smaller test size since we have limited data per class
        test_size = min(0.15, 3/len(class_counts))  # At least 3 examples per class in test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size,
            random_state=42,
            stratify=y,
            shuffle=True
        )
        
        print(f"\nTraining set size: {len(X_train)} examples")
        print(f"Test set size: {len(X_test)} examples")
        
        # Train the model
        print("\n" + "-"*40)
        print("Training model...")
        print("-"*40)
        classifier.train(X_train, y_train)
        
        # Evaluate on test set
        print("\n" + "-"*40)
        print("Evaluating on test set...")
        print("-"*40)
        classifier.evaluate(X_test, y_test)
        
        # Save the model
        print("\n" + "-"*40)
        print("Saving model and artifacts...")
        print("-"*40)
        model_path, vectorizer_path, details_path = classifier.save(MODEL_DIR)
        
        print("\n" + "="*80)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"\nModel saved to: {model_path}")
        print(f"Vectorizer saved to: {vectorizer_path}")
        print(f"Section details saved to: {details_path}")
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("\nTraining failed. Please check the error message above.")
        return False
    
    return True

if __name__ == "__main__":
    main()
