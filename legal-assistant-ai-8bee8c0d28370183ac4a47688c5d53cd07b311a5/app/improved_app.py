import streamlit as st
import os
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
import base64

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

class BNSApp:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.section_details = {}
        self.recognizer = sr.Recognizer()
        self.load_models()
    
    def load_models(self):
        """Load the trained model and vectorizer"""
        try:
            model_dir = Path('saved_models')
            model_path = model_dir / 'bns_classifier.joblib'
            vectorizer_path = model_dir / 'bns_vectorizer.joblib'
            details_path = model_dir / 'bns_section_details.csv'
            
            if not all(p.exists() for p in [model_path, vectorizer_path, details_path]):
                st.error("Model files not found. Please train the model first.")
                return False
            
            with st.spinner('Loading BNS classifier...'):
                self.model = joblib.load(model_path)
                self.vectorizer = joblib.load(vectorizer_path)
                self.section_details = pd.read_csv(details_path, index_col=0).to_dict('index')
            
            return True
            
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False
    
    def predict_section(self, text):
        """Predict the BNS section for the given text"""
        if not text.strip():
            return None, 0.0
        
        try:
            X = self.vectorizer.transform([text])
            predicted_class = self.model.predict(X)[0]
            confidence = np.max(self.model.predict_proba(X))
            return predicted_class, confidence
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            return None, 0.0
    
    def record_voice(self):
        """Record voice input and convert to text"""
        with sr.Microphone() as source:
            st.info("Listening... Speak now")
            try:
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                text = self.recognizer.recognize_google(audio)
                return True, text
            except Exception as e:
                return False, str(e)
    
    def text_to_speech(self, text):
        """Convert text to speech"""
        try:
            tts = gTTS(text=text, lang='en')
            audio_bytes = BytesIO()
            tts.write_to_fp(audio_bytes)
            audio_bytes.seek(0)
            return audio_bytes
        except Exception as e:
            st.error(f"Error generating speech: {str(e)}")
            return None
    
    def display_section(self, section_id, confidence=None):
        """Display section details"""
        if ' - ' in section_id:
            section_id = section_id.split(' - ')[0]
        
        details = self.section_details.get(section_id, {
            'title': 'Section not found',
            'description': 'No description available',
            'punishment': 'No punishment information',
            'example_incidents': ''
        })
        
        st.markdown(f"### BNS Section {section_id}")
        st.markdown(f"**{details['title']}")
        
        with st.expander("View Details"):
            st.markdown(f"**Description:** {details['description']}")
            st.markdown(f"**Punishment:** {details['punishment']}")
            
            if details.get('example_incidents'):
                st.markdown("**Examples:**")
                for ex in details['example_incidents'].split(';'):
                    if ex.strip():
                        st.markdown(f"- {ex.strip()}")
            
            if st.button(f"🔊 Listen to Section {section_id}"):
                audio = self.text_to_speech(f"{details['title']}. {details['description']} Punishment: {details['punishment']}")
                if audio:
                    st.audio(audio, format='audio/mp3')
        
        if confidence:
            st.write(f"Confidence: {confidence:.1%}")
        
        st.markdown("---")
    
    def run(self):
        """Run the Streamlit app"""
        st.set_page_config(
            page_title="BNS Section Finder",
            page_icon="⚖️",
            layout="wide"
        )
        
        # Sidebar
        st.sidebar.title("⚖️ BNS Section Finder")
        page = st.sidebar.radio("Navigation", ["Home", "Find Section", "Browse Sections"])
        
        # Home Page
        if page == "Home":
            st.title("Welcome to BNS Section Finder")
            st.markdown("""
            This application helps you find relevant Bharatiya Nyaya Sanhita (BNS) sections 
            based on incident descriptions.
            
            ### Features:
            - 🔍 Find BNS sections by describing an incident
            - 🎙️ Voice input support
            - 📚 Browse all BNS sections
            - 🔊 Listen to section details
            
            ### How to use:
            1. Go to **Find Section** and describe your incident
            2. Use voice input if preferred
            3. Browse through suggested BNS sections
            4. Listen to section details with audio
            """)
        
        # Find Section Page
        elif page == "Find Section":
            st.title("🔍 Find BNS Section")
            
            # Input method selection
            input_method = st.radio("Choose input method:", ["Text", "Voice"])
            
            if input_method == "Text":
                text = st.text_area("Describe the incident:", height=150)
                if st.button("Find Section"):
                    if text.strip():
                        with st.spinner("Finding relevant BNS section..."):
                            section, confidence = self.predict_section(text)
                            if section:
                                st.success(f"Found relevant BNS section: {section}")
                                self.display_section(section, confidence)
                            else:
                                st.warning("Could not determine a relevant BNS section.")
            
            else:  # Voice input
                if st.button("🎤 Start Recording"):
                    success, result = self.record_voice()
                    if success:
                        st.text_area("Recognized Text:", value=result, height=100)
                        with st.spinner("Finding relevant BNS section..."):
                            section, confidence = self.predict_section(result)
                            if section:
                                st.success(f"Found relevant BNS section: {section}")
                                self.display_section(section, confidence)
                            else:
                                st.warning("Could not determine a relevant BNS section.")
                    else:
                        st.error(f"Error: {result}")
        
        # Browse Sections Page
        elif page == "Browse Sections":
            st.title("📚 Browse BNS Sections")
            
            search_query = st.text_input("Search sections:", "")
            
            # Filter sections
            sections = []
            for section_id, details in self.section_details.items():
                search_text = f"{section_id} {details.get('title', '')} {details.get('description', '')}".lower()
                if not search_query or search_query.lower() in search_text:
                    sections.append((section_id, details.get('title', '')))
            
            sections.sort(key=lambda x: int(x[0]) if x[0].isdigit() else float('inf'))
            
            for section_id, title in sections:
                self.display_section(f"{section_id} - {title}")

if __name__ == "__main__":
    app = BNSApp()
    if app.model is not None and app.vectorizer is not None:
        app.run()
    else:
        st.error("Failed to load the BNS classifier. Please check if the model files are available.")
