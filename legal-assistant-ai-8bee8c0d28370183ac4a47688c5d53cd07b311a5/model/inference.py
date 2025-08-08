import os
import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re
import tempfile
from PIL import Image
import io
import sys

# Add the model directory to the path to allow relative imports
sys.path.append(str(Path(__file__).parent))

# Initialize CLIP processor
IMAGE_PROCESSING_AVAILABLE = False
IMAGE_PROCESSING_ERROR = ""

# Initialize CLIP processor
try:
    from clip_processor import clip_processor
    if clip_processor.model is not None:
        IMAGE_PROCESSING_AVAILABLE = True
except ImportError as e:
    IMAGE_PROCESSING_ERROR = f"Missing required dependencies: {str(e)}. Please install them using: pip install -r requirements.txt"
    print(IMAGE_PROCESSING_ERROR)
except Exception as e:
    IMAGE_PROCESSING_ERROR = f"Error initializing CLIP processor: {str(e)}"
    print(IMAGE_PROCESSING_ERROR)

class LegalAssistantModel:
    def __init__(self, model_path='saved_models'):
        """Initialize the model, vectorizer and load BNS data."""
        # Load BNS sections data
        bns_data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'bns_sections_final.csv')
        if not os.path.exists(bns_data_path):
            raise FileNotFoundError(f"BNS sections file not found at {bns_data_path}")
            
        self.bns_df = pd.read_csv(bns_data_path)
        
        # Initialize vectorizer with better parameters for legal text
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams
            min_df=2,  # Ignore terms that appear in only 1 document
            max_df=0.85  # Ignore terms that appear in >85% of documents
        )
        
        # Prepare text for vectorization
        self.bns_texts = (self.bns_df['section_title'] + ' ' + self.bns_df['content']).fillna('').tolist()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.bns_texts)
    
    def _preprocess_query(self, query):
        """Preprocess the query text for better matching"""
        if not isinstance(query, str):
            return ""
            
        # Convert to lowercase
        query = query.lower()
        # Remove special characters but keep spaces and basic punctuation
        query = re.sub(r'[^\w\s,.!?]', ' ', query)
        # Remove extra whitespace
        query = ' '.join(query.split())
        return query
        
    def _clean_section_text(self, text):
        """Clean up section text by removing unwanted patterns and formatting."""
        if not text or pd.isna(text):
            return ""
            
        # Convert to string in case it's not
        text = str(text)
        
        # Remove lines with only separators (dashes, underscores, etc.)
        lines = [line for line in text.split('\n') 
                if not all(c in ' _-=' for c in line.strip())]
        
        # Remove page numbers, section markers, and other artifacts
        cleaned_lines = []
        for line in lines:
            # Remove page numbers and section markers
            line = re.sub(r'\b(?:Sec\.?\s*\d+\]?|Page\s*\d+|\d+\s*\])\s*', '', line)
            # Remove chapter headers
            line = re.sub(r'CHAPTER\s*[IVXLCDM]+\s*OF\s*[A-Z\s]+', '', line, flags=re.IGNORECASE)
            # Remove multiple spaces and trim
            line = ' '.join(line.split())
            if line.strip():
                cleaned_lines.append(line.strip())
        
        # Join with single newlines and remove any remaining excessive whitespace
        cleaned_text = '\n'.join(cleaned_lines)
        cleaned_text = ' '.join(cleaned_text.split())
        
        # Remove any remaining artifacts
        cleaned_text = re.sub(r'\s*[_-]{3,}\s*', ' ', cleaned_text)  # Remove separator lines
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()  # Normalize whitespace
        
        return cleaned_text

    def find_most_similar_sections(self, query, top_n=5, is_image_analysis=False):
        """Find the most similar BNS sections to the query or image analysis"""
        # Preprocess the query
        query = self._preprocess_query(query)
        
        # Transform query to TF-IDF vector
        query_vec = self.vectorizer.transform([query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get top N most similar sections
        top_indices = similarities.argsort()[-top_n * 2:][::-1]  # Get extra sections for filtering
        
        # Prepare results
        results = []
        seen_sections = set()
        
        for idx in top_indices:
            if len(results) >= top_n:
                break
                
            if similarities[idx] > 0.1:  # Threshold for relevance
                section_num = str(self.bns_df.iloc[idx]['section_number']).strip()
                
                # Skip if we've already seen this section (duplicate check)
                if section_num in seen_sections:
                    continue
                    
                seen_sections.add(section_num)
                
                # Clean the content
                raw_content = str(self.bns_df.iloc[idx]['content'])
                cleaned_content = self._clean_section_text(raw_content)
                
                # Only include if we have valid content
                if not cleaned_content or len(cleaned_content) < 10:  # Skip very short sections
                    continue
                    
                result = {
                    'section_number': section_num,
                    'section_title': str(self.bns_df.iloc[idx]['section_title']).strip(),
                    'description': cleaned_content,
                    'score': float(similarities[idx]),
                    'punishment': self._extract_punishment(raw_content),
                    'is_image_analysis': is_image_analysis
                }
                
                # For image analysis, add additional metadata
                if is_image_analysis:
                    result['scene_description'] = query
                    
                results.append(result)
        
        # Sort by score (descending)
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_n]

    def analyze_image(self, image_file):
        """Analyze an image and return relevant legal information using CLIP."""
        from clip_processor import clip_processor
        
        if not IMAGE_PROCESSING_AVAILABLE or clip_processor.model is None:
            error_message = 'Image processing is not available in the current configuration.\n\n'
            if IMAGE_PROCESSING_ERROR:
                error_message += f'Error details: {IMAGE_PROCESSING_ERROR}\n\n'
            error_message += 'Please ensure all required dependencies are installed. Run this command and restart the app:\n\n'
            error_message += '```\n'
            error_message += 'pip install -r requirements.txt\n'
            error_message += '```'
            
            return {
                'error': error_message,
                'success': False
            }
        
        try:
            # Save the uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                if hasattr(image_file, 'read'):
                    tmp_file.write(image_file.read())
                else:
                    with open(image_file, 'rb') as f:
                        tmp_file.write(f.read())
                tmp_file_path = tmp_file.name
            
            # Analyze the image using CLIP
            try:
                analysis = clip_processor.analyze_scene(tmp_file_path)
            except Exception as e:
                return {
                    'error': f'Error analyzing image with CLIP: {str(e)}',
                    'success': False
                }
            
            if not analysis or 'scene_description' not in analysis:
                return {
                    'error': 'Failed to analyze the image. Please try again with a different image.',
                    'success': False
                }
            
            # Find relevant BNS sections based on the scene description
            relevant_sections = self.find_most_similar_sections(
                analysis['scene_description'], 
                top_n=3,
                is_image_analysis=True
            )
            
            # Clean up the temporary file
            try:
                os.unlink(tmp_file_path)
            except Exception as e:
                print(f"Warning: Could not delete temporary file: {e}")
            
            return {
                'success': True,
                'scene_description': analysis['scene_description'],
                'detected_objects': analysis.get('detected_objects', []),
                'detected_text': analysis.get('detected_text', []),
                'sections': relevant_sections,
                'suggested_actions': [
                    'Document the scene with timestamps if possible.',
                    'Contact legal authorities if you believe a crime has been committed.',
                    'Consult with a legal professional for specific advice.'
                ]
            }
            
        except Exception as e:
            # Clean up the temporary file in case of error
            try:
                if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
            except:
                pass
                
            return {
                'error': f'Error processing image: {str(e)}',
                'success': False
            }
    
    def _extract_punishment(self, content):
        """Extract and clean punishment information from section content."""
        if not content or pd.isna(content):
            return 'Not specified'
            
        # Clean up the content first
        content = str(content)
        
        # Remove lines with only underscores or other separators
        content = '\n'.join([line for line in content.split('\n') 
                            if not all(c in ' _-=' for c in line.strip())])
        
        # Remove page numbers and section markers
        content = re.sub(r'\b(?:Sec\.?\s*\d+\]?|Page\s*\d+|\d+\s*\])\s*', '', content)
        
        # Remove chapter headers
        content = re.sub(r'CHAPTER\s*[IVXLCDM]+\s*OF\s*[A-Z\s]+', '', content, flags=re.IGNORECASE)
        
        # Remove extra whitespace and newlines
        content = ' '.join(content.split())
        
        # Look for punishment patterns
        punishment_patterns = [
            r'shall be punished with (.*?)(?=\.|$)',
            r'punishable with (.*?)(?=\.|$)',
            r'shall be liable to (.*?)(?=\.|$)'
        ]
        
        for pattern in punishment_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                punishment = match.group(1).strip()
                # Clean up any remaining unwanted text
                punishment = re.sub(r'\b(?:Provided that|Explanation|Illustration).*', '', punishment, flags=re.IGNORECASE)
                return punishment.strip()
        
        # If no specific pattern found, look for punishment indicators
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
