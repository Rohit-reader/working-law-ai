import os
import sys
import streamlit as st
import speech_recognition as sr
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import spacy
from gtts import gTTS
from io import BytesIO
import base64

# Add parent directory to path to import model
sys.path.append(str(Path(__file__).parent.parent))

# Set page config
st.set_page_config(
    page_title="BNS Section Finder",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stTextInput>div>div>input {
        padding: 12px !important;
        border-radius: 8px !important;
        border: 1px solid #ced4da !important;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .section-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 5px solid #4a6fa5;
    }
    .section-title {
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    .section-punishment {
        background: #f8f9fa;
        padding: 0.75rem;
        border-radius: 6px;
        margin: 0.75rem 0;
        font-size: 0.9rem;
    }
    .example-incident {
        font-style: italic;
        color: #6c757d;
        margin: 0.5rem 0;
        padding: 0.5rem;
        background: #f8f9fa;
        border-left: 3px solid #dee2e6;
    }
    .confidence {
        font-size: 0.85rem;
        color: #6c757d;
        margin-top: 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        padding: 0 16px;
        border-radius: 8px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e9ecef;
    }
    </style>
""", unsafe_allow_html=True)

class BNSClassifierApp:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.section_details = {}
        self.nlp = spacy.load('en_core_web_sm')
        self.recognizer = sr.Recognizer()
        self.load_models()
    
    def load_models(self):
        """Load the trained model and vectorizer"""
        try:
            model_dir = os.path.join('saved_models')
            model_path = os.path.join(model_dir, 'enhanced_bns_classifier.joblib')
            vectorizer_path = os.path.join(model_dir, 'enhanced_bns_vectorizer.joblib')
            details_path = os.path.join(model_dir, 'bns_section_details_enhanced.csv')
            
            if not all(os.path.exists(p) for p in [model_path, vectorizer_path, details_path]):
                st.error("Model files not found. Please train the model first.")
                return False
            
            with st.spinner('Loading BNS classifier...'):
                self.model = joblib.load(model_path)
                self.vectorizer = joblib.load(vectorizer_path)
                
                # Load section details
                details_df = pd.read_csv(details_path, index_col=0)
                self.section_details = details_df.to_dict('index')
                
                # Extract section numbers from the index
                self.section_numbers = sorted([str(k) for k in self.section_details.keys()])
                
            return True
            
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False
    
    def preprocess_text(self, text):
        """Preprocess the input text"""
        if not text or not isinstance(text, str):
            return ""
        
        # Basic cleaning
        text = text.lower().strip()
        
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
    
    def predict_section(self, text):
        """Predict the BNS section for the given text"""
        if not text.strip():
            return None, 0.0
        
        try:
            # Preprocess the input text
            processed_text = self.preprocess_text(text)
            
            # Vectorize the text
            X = self.vectorizer.transform([processed_text])
            
            # Get prediction and confidence
            predicted_class = self.model.predict(X)[0]
            confidence = np.max(self.model.decision_function(X))
            
            # Convert confidence to a probability-like score between 0 and 1
            confidence = 1 / (1 + np.exp(-confidence))  # Sigmoid function
            
            return predicted_class, confidence
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            return None, 0.0
    
    def get_section_details(self, section_id):
        """Get details for a specific section"""
        # Extract just the section number if it's in the format '101 - Punishment for murder'
        if ' - ' in section_id:
            section_id = section_id.split(' - ')[0]
        
        return self.section_details.get(section_id, {
            'title': 'Section not found',
            'description': 'No description available',
            'punishment': 'No punishment information available',
            'example_incidents': ''
        })
    
    def text_to_speech(self, text, lang='en'):
        """Convert text to speech and return audio data"""
        try:
            tts = gTTS(text=text, lang=lang, slow=False)
            audio_bytes = BytesIO()
            tts.write_to_fp(audio_bytes)
            audio_bytes.seek(0)
            return audio_bytes
        except Exception as e:
            st.error(f"Error generating speech: {str(e)}")
            return None
    
    def record_voice(self):
        """Record voice input using the microphone"""
        with sr.Microphone() as source:
            st.info("Listening... Speak now")
            audio = self.recognizer.listen(source)
            
            try:
                text = self.recognizer.recognize_google(audio)
                return text
            except sr.UnknownValueError:
                st.warning("Could not understand audio")
                return ""
            except sr.RequestError as e:
                st.error(f"Could not request results; {str(e)}")
                return ""
    
    def display_section_card(self, section_id, confidence=None):
        """Display a card with section details"""
        details = self.get_section_details(section_id)
        
        # Create a card for the section
        card = f"""
        <div class="section-card">
            <h3 class="section-title">BNS Section {section_id}</h3>
            <p><strong>Title:</strong> {details.get('title', 'N/A')}</p>
            <div class="section-punishment">
                <strong>Punishment:</strong> {details.get('punishment', 'No punishment information available')}
            </div>
            <p><strong>Description:</strong> {details.get('description', 'No description available')}</p>
            {self._format_examples(details.get('example_incidents', ''))}
        </div>
        """
        
        if confidence is not None:
            confidence_html = f"""
            <div class="confidence">
                <strong>Confidence:</strong> {confidence:.1%}
            </div>
            """
            card = card.replace("</div>", f"{confidence_html}</div>")
        
        st.markdown(card, unsafe_allow_html=True)
        
        # Add text-to-speech button
        if st.button(f"🔊 Listen to Section {section_id}"):
            text_to_speak = f"""
            BNS Section {section_id}. {details.get('title', '')}.
            {details.get('description', '')}
            Punishment: {details.get('punishment', '')}
            """
            audio_data = self.text_to_speech(text_to_speak)
            if audio_data:
                audio_base64 = base64.b64encode(audio_data.read()).decode('utf-8')
                audio_html = f"""
                <audio autoplay="true" controls style="width: 100%;">
                    <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                    Your browser does not support the audio element.
                </audio>
                """
                st.markdown(audio_html, unsafe_allow_html=True)
    
    def _format_examples(self, examples_str):
        """Format example incidents as HTML"""
        if not examples_str or not isinstance(examples_str, str):
            return ""
        
        examples = [ex.strip() for ex in examples_str.split(';') if ex.strip()]
        if not examples:
            return ""
        
        examples_html = "<div style='margin-top: 1rem;'><strong>Example Incidents:</strong>"
        for ex in examples:
            examples_html += f"<div class='example-incident'>{ex}</div>"
        examples_html += "</div>"
        
        return examples_html
    
    def run(self):
        """Run the Streamlit app"""
        st.title("⚖️ BNS Section Finder")
        st.markdown("""
            Find the relevant Bharatiya Nyaya Sanhita (BNS) section for your legal incident. 
            Enter a description of the incident below or use the voice input feature.
        """)
        
        # Initialize session state for voice recording
        if 'listening' not in st.session_state:
            st.session_state.listening = False
        
        # Create tabs for different functionalities
        tab1, tab2 = st.tabs(["Find BNS Section", "Browse All Sections"])
        
        with tab1:
            # Input section
            col1, col2 = st.columns([4, 1])
            with col1:
                incident_desc = st.text_area(
                    "Describe the incident in detail:",
                    placeholder="E.g., A person intentionally killed another person during a robbery...",
                    height=150
                )
            
            with col2:
                st.write("Or")
                if st.button("🎤 Record Voice", use_container_width=True):
                    st.session_state.listening = True
                
                if st.session_state.listening:
                    voice_text = self.record_voice()
                    if voice_text:
                        incident_desc = voice_text
                        st.session_state.listening = False
                    else:
                        st.session_state.listening = False
            
            # Predict button
            if st.button("Find Relevant BNS Section", type="primary"):
                if not incident_desc.strip():
                    st.warning("Please enter or record a description of the incident.")
                else:
                    with st.spinner("Analyzing incident and finding relevant BNS section..."):
                        predicted_section, confidence = self.predict_section(incident_desc)
                        
                        if predicted_section:
                            st.success(f"Found relevant BNS section: {predicted_section}")
                            self.display_section_card(predicted_section, confidence)
                        else:
                            st.warning("Could not determine a relevant BNS section. Please try with a more detailed description.")
            
            # Show recent predictions (if any)
            if 'recent_predictions' not in st.session_state:
                st.session_state.recent_predictions = []
            
            if incident_desc.strip() and len(st.session_state.recent_predictions) > 0:
                st.subheader("Recent Predictions")
                for section_id in st.session_state.recent_predictions[-3:]:  # Show last 3
                    self.display_section_card(section_id)
        
        with tab2:
            st.subheader("Browse All BNS Sections")
            
            # Search and filter section
            search_query = st.text_input("Search sections by keyword:", "")
            
            # Filter sections based on search query
            filtered_sections = []
            for section_id in self.section_numbers:
                details = self.section_details.get(section_id, {})
                search_text = f"{section_id} {details.get('title', '')} {details.get('description', '')} {details.get('punishment', '')}".lower()
                if not search_query or search_query.lower() in search_text:
                    filtered_sections.append(section_id)
            
            if not filtered_sections:
                st.info("No sections match your search criteria.")
            else:
                # Display sections in a grid
                cols = st.columns(2)  # Two columns for better layout
                for i, section_id in enumerate(filtered_sections):
                    with cols[i % 2]:
                        self.display_section_card(section_id)
                    
                    # Add a separator between rows
                    if i % 2 == 1 and i < len(filtered_sections) - 1:
                        st.markdown("<hr style='margin: 1.5rem 0; border: 0.5px solid #dee2e6;'/>", unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    app = BNSClassifierApp()
    if app.model is not None and app.vectorizer is not None:
        app.run()
    else:
        st.error("Failed to load the BNS classifier. Please make sure the model files are available.")
