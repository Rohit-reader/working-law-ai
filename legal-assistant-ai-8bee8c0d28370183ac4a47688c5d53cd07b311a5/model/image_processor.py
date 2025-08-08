import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import easyocr
import cv2
import numpy as np
from transformers import AutoModelForObjectDetection, AutoFeatureExtractor

class ImageProcessor:
    def __init__(self):
        # Initialize the model and processor
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = "hustvl/yolos-tiny"  # Lightweight model for CPU
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
        self.model = AutoModelForObjectDetection.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        
        # Initialize EasyOCR for text detection
        self.reader = easyocr.Reader(['en'])
        
        # Common objects to detect (can be expanded)
        self.legal_objects = [
            'knife', 'gun', 'weapon', 'drug', 'money', 'person', 'vehicle',
            'bottle', 'phone', 'computer', 'document', 'knives', 'firearm'
        ]
        
    def detect_objects(self, image_path):
        """Detect objects in the image using YOLOS model"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            inputs = self.feature_extractor(images=image, return_tensors="pt").to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Process outputs
            target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
            results = self.feature_extractor.post_process_object_detection(
                outputs, threshold=0.5, target_sizes=target_sizes
            )[0]
            
            # Get detected objects
            detected_objects = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                if score > 0.5:  # Confidence threshold
                    label_name = self.model.config.id2label[label.item()]
                    detected_objects.append({
                        'label': label_name,
                        'score': round(score.item(), 3),
                        'box': [round(i, 2) for i in box.tolist()]
                    })
            
            return detected_objects
            
        except Exception as e:
            print(f"Error in object detection: {str(e)}")
            return []
    
    def extract_text(self, image_path):
        """Extract text from the image using EasyOCR"""
        try:
            # Read image
            image = cv2.imread(image_path)
            # Convert to RGB (EasyOCR expects RGB)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Extract text
            results = self.reader.readtext(image)
            return [result[1] for result in results]  # Return only the text
        except Exception as e:
            print(f"Error in text extraction: {str(e)}")
            return []
    
    def analyze_scene(self, image_path):
        """Analyze the image and return a description of the scene"""
        # Detect objects
        objects = self.detect_objects(image_path)
        
        # Extract text
        text = self.extract_text(image_path)
        
        # Filter for legal-relevant objects
        relevant_objects = [
            obj for obj in objects 
            if any(legal_obj in obj['label'].lower() for legal_obj in self.legal_objects)
        ]
        
        # Create a description
        scene_description = "Scene contains: "
        if relevant_objects:
            scene_description += ", ".join([obj['label'] for obj in relevant_objects])
        else:
            scene_description += "no clearly identifiable legal objects"
        
        detected_text = []
        if text:
            scene_description += ". Text detected: " + " | ".join(text)
            detected_text = text
        
        return {
            'scene_description': scene_description,
            'detected_objects': relevant_objects,
            'detected_text': detected_text,
            'scene_description': scene_description,
            'detected_objects': relevant_objects,
            'detected_text': text
        }

# Singleton instance
image_processor = ImageProcessor()
