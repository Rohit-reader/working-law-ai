import os
import pandas as pd
import numpy as np
import joblib
import spacy
import re
import random
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
import warnings
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

class BNSTextClassifier:
    def __init__(self, model_dir='saved_models'):
        """Initialize the BNS text classifier"""
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize NLP pipeline
        self.nlp = spacy.load('en_core_web_sm')
        
        # Initialize model components
        self.vectorizer = TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 3),
            stop_words='english',
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
        
        self.classifier = LinearSVC(
            class_weight='balanced',
            random_state=42,
            max_iter=5000,
            C=0.5,
            dual=False
        )
        
        self.calibrated_classifier = None
        self.section_details = {}
        self.classes_ = None
    
    def preprocess_text(self, text):
        """Preprocess the input text"""
        if not text or not isinstance(text, str):
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
    
    def augment_text(self, text, n_augment=2):
        """Generate augmented versions of the text"""
        if not text or not isinstance(text, str):
            return []
            
        augmented_texts = []
        words = text.split()
        
        # Only augment if we have enough words
        if len(words) < 4:
            return [text]
        
        for _ in range(n_augment):
            # Randomly shuffle words (but keep the meaning somewhat intact)
            if random.random() > 0.5 and len(words) > 3:
                # Keep first and last words in place, shuffle the middle
                first_word = words[0]
                last_word = words[-1]
                middle = words[1:-1]
                random.shuffle(middle)
                new_text = ' '.join([first_word] + middle + [last_word])
                augmented_texts.append(new_text)
            
            # Randomly drop some words (but not too many)
            if random.random() > 0.7 and len(words) > 5:
                n_drop = min(2, len(words) // 4)
                drop_indices = sorted(random.sample(range(len(words)), len(words) - n_drop))
                new_text = ' '.join([words[i] for i in drop_indices])
                augmented_texts.append(new_text)
        
        return list(set(augmented_texts))  # Remove duplicates
    
    def load_data(self, data_path):
        """Load and preprocess the BNS dataset with data augmentation"""
        # Load the dataset
        df = pd.read_csv(data_path)
        df = df.dropna(subset=['description', 'example_incidents', 'punishment'])
        
        # Store section details for later use
        self.section_details = df.set_index('section').to_dict('index')
        
        texts = []
        labels = []
        
        print("Processing sections and augmenting data...")
        for _, row in tqdm(df.iterrows(), total=len(df)):
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
            
            # Add keywords as additional context
            if pd.notna(row.get('keywords', '')):
                section_examples.append(f"Keywords: {row['keywords']}")
            
            # Process and augment the examples
            for example in section_examples:
                # Add the original example
                processed = self.preprocess_text(example)
                if processed:
                    texts.append(processed)
                    labels.append(label)
                
                # Generate augmented versions
                augmented = self.augment_text(example)
                for aug_text in augmented:
                    processed_aug = self.preprocess_text(aug_text)
                    if processed_aug and processed_aug != processed:  # Don't add duplicates
                        texts.append(processed_aug)
                        labels.append(label)
        
        return texts, labels, df
    
    def train(self, X, y, test_size=0.15):
        """Train the classifier with cross-validation and calibration"""
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size,
            random_state=42,
            stratify=y,
            shuffle=True
        )
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        # Vectorize the training data
        print("Fitting vectorizer...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        
        # Calculate class weights
        print("Calculating class weights...")
        classes = np.unique(y_train)
        self.classes_ = classes
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, class_weights))
        
        # Update classifier with balanced class weights
        self.classifier.class_weight = class_weight_dict
        
        # Perform cross-validation
        print("\nPerforming cross-validation...")
        cv = StratifiedKFold(n_splits=min(5, len(classes)), shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            self.classifier, X_train_vec, y_train,
            cv=cv,
            scoring='f1_weighted',
            n_jobs=-1
        )
        
        print(f"Cross-validation F1 scores: {cv_scores}")
        print(f"Mean CV F1 score: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
        
        # Train the final model on the full training set
        print("\nTraining final model...")
        self.classifier.fit(X_train_vec, y_train)
        
        # Calibrate the classifier for better probability estimates
        print("Calibrating classifier...")
        self.calibrated_classifier = CalibratedClassifierCV(
            self.classifier, 
            cv='prefit',
            method='sigmoid'
        )
        self.calibrated_classifier.fit(X_train_vec, y_train)
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        X_test_vec = self.vectorizer.transform(X_test)
        y_pred = self.calibrated_classifier.predict(X_test_vec)
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        print("\nAccuracy:", accuracy_score(y_test, y_pred))
        print("Weighted F1:", f1_score(y_test, y_pred, average='weighted', zero_division=0))
        
        return X_test, y_test
    
    def save_model(self):
        """Save the model and related artifacts"""
        model_path = self.model_dir / 'bns_classifier.joblib'
        vectorizer_path = self.model_dir / 'bns_vectorizer.joblib'
        details_path = self.model_dir / 'bns_section_details.csv'
        
        # Save the model components
        joblib.dump(self.calibrated_classifier, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        
        # Save section details as a DataFrame
        details_df = pd.DataFrame.from_dict(self.section_details, orient='index')
        details_df.to_csv(details_path)
        
        print(f"\nModel saved to {model_path}")
        print(f"Vectorizer saved to {vectorizer_path}")
        print(f"Section details saved to {details_path}")
        
        return model_path, vectorizer_path, details_path
    
    @classmethod
    def load_model(cls, model_dir='saved_models'):
        """Load a trained model"""
        model_path = Path(model_dir) / 'bns_classifier.joblib'
        vectorizer_path = Path(model_dir) / 'bns_vectorizer.joblib'
        details_path = Path(model_dir) / 'bns_section_details.csv'
        
        if not all(p.exists() for p in [model_path, vectorizer_path, details_path]):
            raise FileNotFoundError("One or more model files not found. Please train the model first.")
        
        # Create a new instance
        classifier = cls(model_dir)
        
        # Load the components
        classifier.calibrated_classifier = joblib.load(model_path)
        classifier.classifier = classifier.calibrated_classifier.base_estimator
        classifier.vectorizer = joblib.load(vectorizer_path)
        
        # Load section details
        details_df = pd.read_csv(details_path, index_col=0)
        classifier.section_details = details_df.to_dict('index')
        classifier.classes_ = np.array([f"{idx} - {val['title']}" for idx, val in classifier.section_details.items()])
        
        return classifier

def main():
    # Configuration
    DATA_PATH = os.path.join('data', 'raw', 'bns_enhanced_dataset.csv')
    MODEL_DIR = 'saved_models'
    
    # Create output directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print("=" * 80)
    print("BNS Section Classifier Training")
    print("=" * 80)
    
    # Initialize and train the classifier
    classifier = BNSTextClassifier(MODEL_DIR)
    
    try:
        # Load and preprocess data
        print("\nLoading and preprocessing data...")
        X, y, section_details = classifier.load_data(DATA_PATH)
        
        # Print class distribution
        print("\nClass distribution in dataset:")
        print(pd.Series(y).value_counts().sort_index())
        
        # Train the model
        X_test, y_test = classifier.train(X, y)
        
        # Save the model
        print("\nSaving model and artifacts...")
        classifier.save_model()
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("\nTraining failed. Please check the error message above.")
        return False
    
    return True

if __name__ == "__main__":
    main()
