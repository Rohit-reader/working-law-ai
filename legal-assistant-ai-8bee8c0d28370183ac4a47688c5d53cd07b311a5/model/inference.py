import os
import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class LegalAssistantModel:
    def __init__(self, model_path='saved_models'):
        """Initialize the model, vectorizer and load BNS data."""
        # Load BNS sections data
        bns_data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'bns_sections_final.csv')
        if not os.path.exists(bns_data_path):
            raise FileNotFoundError(f"BNS sections file not found at {bns_data_path}")
            
        self.bns_df = pd.read_csv(bns_data_path)
        
        # Initialize TF-IDF vectorizer for similarity matching
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
        # Fit vectorizer on BNS section titles and content
        self.bns_texts = (self.bns_df['section_title'] + ' ' + self.bns_df['content']).fillna('').tolist()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.bns_texts)
    
    def find_most_similar_sections(self, query, top_n=5):
        """Find the most relevant BNS sections to the query with focus on punishment."""
        # Check if query is specifically about punishment
        punishment_keywords = ['punish', 'sentence', 'penalty', 'jail', 'imprison', 'fine', 'death', 'life']
        is_punishment_query = any(keyword in query.lower() for keyword in punishment_keywords)
        
        # Transform query to TF-IDF
        query_vec = self.vectorizer.transform([query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get all sections with similarity above threshold
        relevant_indices = [i for i, score in enumerate(similarities) if score > 0.1]
        
        # Prepare results with scores
        results = []
        for idx in relevant_indices:
            section = self.bns_df.iloc[idx]
            punishment = section.get('punishment', '')
            
            # Skip if no punishment information and query is about punishment
            if is_punishment_query and (not punishment or pd.isna(punishment)):
                continue
                
            # Calculate score - boost if punishment matches query
            score = float(similarities[idx])
            if is_punishment_query and punishment and pd.notna(punishment):
                score *= 1.5  # Boost score for punishment-related queries
            
            results.append({
                'section_number': section['section_number'],
                'section_title': section['section_title'],
                'description': section['content'],  # Using 'content' as description
                'punishment': self._extract_punishment(section['content']),  # Extract punishment from content
                'score': score
            })
        
        # Sort by score (descending) and take top N
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_n]
        
        return results

    def _extract_punishment(self, content):
        """Extract punishment information from section content."""
        if not content or pd.isna(content):
            return 'Not specified'
        
        # Look for common punishment indicators in the content
        punishment_indicators = [
            'punish', 'imprisonment', 'fine', 'death', 'life', 'penalty',
            'sentence', 'jail', 'rigorous', 'simple', 'extend', 'years',
            'months', 'rupees', 'both', 'shall be punished', 'liable'
        ]
        
        # Find the punishment part of the content (usually after 'punishable with' or similar)
        content_lower = content.lower()
        punishment_start = -1
        
        for indicator in punishment_indicators:
            idx = content_lower.find(indicator)
            if idx > 0 and (punishment_start == -1 or idx < punishment_start):
                punishment_start = idx
        
        if punishment_start > 0:
            # Get the punishment part and clean it up
            punishment = content[punishment_start:]
            # Remove any trailing period if it's not part of an abbreviation
            if punishment.endswith('.'):
                punishment = punishment[:-1]
            return punishment.strip()
        
        return 'Not specified'

def load_model(model_path='saved_models'):
    """Load the trained model and vectorizer."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory not found at {model_path}")
    
    return LegalAssistantModel(model_path)

def get_legal_advice(text, model=None):
    """Get legal advice based on the input text with focus on punishment details."""
    if model is None:
        model = load_model()
    
    # Find relevant BNS sections with increased top_n for better filtering
    relevant_sections = model.find_most_similar_sections(text, top_n=5)
    
    # Filter to only include sections with punishment details if query is punishment-related
    punishment_keywords = ['punish', 'sentence', 'penalty', 'jail', 'imprison', 'fine', 'death', 'life']
    is_punishment_query = any(keyword in text.lower() for keyword in punishment_keywords)
    
    if is_punishment_query:
        relevant_sections = [s for s in relevant_sections 
                           if s['punishment'] and s['punishment'] != 'Not specified']
    
    # Prepare response
    response = {
        'input_text': text,
        'relevant_sections': relevant_sections[:3],  # Return top 3 most relevant
        'query_type': 'punishment' if is_punishment_query else 'general',
        'suggested_actions': [
            'Consult with a lawyer for specific legal advice',
            'Review the full text of relevant BNS sections',
            'Document all relevant details and evidence',
            'Note: Punishment details are for reference only and may vary case by case'
        ]
    }
    
    return response

if __name__ == "__main__":
    # Example usage
    test_text = "I was injured at work and my employer is not providing compensation."
    result = get_legal_advice(test_text)
    print("Legal Assistance Result:")
    print(f"Input: {result['input_text']}")
    print(f"Category: {result['predicted_category']}")
    print("Suggested Actions:")
    for action in result['suggested_actions']:
        print(f"- {action}")
