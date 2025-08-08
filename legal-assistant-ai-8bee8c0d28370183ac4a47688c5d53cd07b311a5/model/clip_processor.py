import torch
import clip
from PIL import Image
import numpy as np

class CLIPProcessor:
    def __init__(self):
        """Initialize the CLIP model and preprocessing."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = None, None
        self.legal_concepts = [
            "weapon", "violence", "theft", "fraud", "assault", "vandalism", 
            "drugs", "alcohol", "traffic violation", "cyber crime", 
            "property damage", "harassment", "threat", "forgery", "bribery",
            "document", "signature", "license plate", "weapon", "suspicious package",
            "person", "vehicle", "building", "public place", "private property"
        ]
        self._load_model()
    
    def _load_model(self):
        """Load the CLIP model and preprocessing."""
        try:
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            self.model.eval()
            return True
        except Exception as e:
            print(f"Error loading CLIP model: {e}")
            return False
    
    def get_image_features(self, image_path):
        """Get image features using CLIP."""
        try:
            image = Image.open(image_path).convert("RGB")
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            
            return image_features
        except Exception as e:
            print(f"Error processing image: {e}")
            return None
    
    def analyze_scene(self, image_path):
        """Analyze an image and return relevant legal concepts."""
        if not self.model:
            return {
                'scene_description': 'CLIP model not loaded',
                'detected_objects': [],
                'detected_text': []
            }
        
        try:
            # Get image features
            image_features = self.get_image_features(image_path)
            if image_features is None:
                return {
                    'scene_description': 'Could not process image',
                    'detected_objects': [],
                    'detected_text': []
                }
            
            # Prepare text inputs
            text_inputs = torch.cat([clip.tokenize(f"a photo of {c}") for c in self.legal_concepts]).to(self.device)
            
            # Calculate similarity
            with torch.no_grad():
                text_features = self.model.encode_text(text_inputs)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarity
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                values, indices = similarity[0].topk(5)  # Get top 5 matches
            
            # Prepare results
            detected_objects = []
            for value, idx in zip(values, indices):
                concept = self.legal_concepts[idx]
                score = float(value) * 100
                if score > 5:  # Only include if confidence > 5%
                    detected_objects.append({
                        'label': concept,
                        'score': round(score, 2)
                    })
            
            # Generate scene description
            if detected_objects:
                scene_desc = "Detected: " + ", ".join([f"{obj['label']} ({obj['score']:.1f}%)" 
                                                    for obj in detected_objects])
            else:
                scene_desc = "No relevant legal concepts detected in the image."
            
            return {
                'scene_description': scene_desc,
                'detected_objects': detected_objects,
                'detected_text': []  # CLIP doesn't do OCR, so we'll leave this empty
            }
            
        except Exception as e:
            print(f"Error in scene analysis: {e}")
            return {
                'scene_description': f'Error analyzing image: {str(e)}',
                'detected_objects': [],
                'detected_text': []
            }

# Singleton instance
clip_processor = CLIPProcessor()
