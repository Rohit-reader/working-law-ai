import streamlit as st
import joblib
import os
import pandas as pd
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.train_bns_classifier import BNSClassifier

# Set page config
st.set_page_config(
    page_title="BNS Section Finder",
    page_icon="⚖️",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        max-width: 1200px;
        padding: 2rem;
    }
    .section-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .section-title {
        color: #1e40af;
        margin-bottom: 0.5rem;
    }
    .section-punishment {
        color: #9c2c13;
        font-weight: 500;
        margin: 0.5rem 0;
    }
    .section-desc {
        color: #374151;
        line-height: 1.6;
    }
    .stTextArea > div > div > textarea {
        min-height: 150px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the BNS classifier model"""
    try:
        classifier = BNSClassifier.load('saved_models')
        section_details = pd.read_csv('saved_models/bns_section_details.csv')
        return classifier, section_details
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

def get_section_details(section, section_details):
    """Get details for a specific BNS section"""
    section_parts = section.split(' - ', 1)
    if len(section_parts) == 2:
        section_num = section_parts[0].strip()
        section_title = section_parts[1].strip()
        details = section_details[section_details['section'] == section_num].iloc[0]
        return {
            'section': section_num,
            'title': section_title,
            'description': details['description'],
            'punishment': details['punishment']
        }
    return None

def main():
    st.title("⚖️ BNS Section Finder")
    st.markdown("Enter the details of a legal incident to find relevant BNS (Bharatiya Nyaya Sanhita) sections.")
    
    # Load model and section details
    classifier, section_details = load_model()
    
    # Input section
    with st.form("incident_form"):
        incident_desc = st.text_area(
            "Describe the incident in detail:",
            placeholder="Example: A person was caught stealing a mobile phone from a shop...",
            help="Provide as much detail as possible about the incident for accurate section matching."
        )
        
        submit_button = st.form_submit_button("Find Relevant BNS Sections")
    
    if submit_button and incident_desc.strip():
        with st.spinner("Analyzing the incident and finding relevant BNS sections..."):
            # Get predictions
            predicted_sections = classifier.predict([incident_desc])
            
            if predicted_sections:
                st.success("Found relevant BNS sections:")
                
                # Display each predicted section with details
                for section in predicted_sections:
                    details = get_section_details(section, section_details)
                    if details:
                        with st.container():
                            st.markdown(f"### {details['section']} - {details['title']}")
                            st.markdown(f"<div class='section-card'>"
                                      f"<div class='section-desc'>{details['description']}</div>"
                                      f"<div class='section-punishment'>Punishment: {details['punishment']}</div>"
                                      f"</div>", unsafe_allow_html=True)
                            st.write("")
            else:
                st.warning("No specific BNS sections matched the provided description. Try being more detailed.")
    
    # Add some example queries
    with st.expander("Example Queries"):
        st.markdown("""
        Try these example queries:
        - "A person was caught stealing a mobile phone from a shop"
        - "A woman was harassed by her husband for dowry"
        - "A person was found guilty of causing death by reckless driving"
        - "Someone forged documents to claim property"
        - "A person was caught taking bribes in a government office"
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        **Note:** This tool is for informational purposes only and does not constitute legal advice. 
        Always consult with a qualified legal professional for specific legal matters.
        """
    )

if __name__ == "__main__":
    main()
