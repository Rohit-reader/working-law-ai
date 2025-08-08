import streamlit as st
import os
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from model.inference import get_legal_advice, load_model
from speech.voice_to_text import VoiceRecognizer

def main():
    # Set page config
    st.set_page_config(
        page_title="Legal Assistant AI",
        page_icon="⚖️",
        layout="wide"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {color: #2c3e50;}
    .sidebar .sidebar-content {background-color: #f8f9fa;}
    .stButton>button {width: 100%;}
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚖️ Legal Assistant AI")
    st.sidebar.markdown("### Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Text Input", "Voice Input", "About"])
    
    # Load model (cached)
    @st.cache_resource
    def load_ai_model():
        return load_model()
    
    # Home Page
    if page == "Home":
        st.markdown("# Welcome to Legal Assistant AI")
        st.markdown("### Your AI-powered legal assistant")
        
        st.markdown("""
        This application helps you get preliminary legal information and guidance.
        
        ### Features:
        - 📝 Text-based legal queries
        - 🎙️ Voice input support
        - ⚡ Quick legal advice
        - 📚 Resource suggestions
        
        ### How to use:
        1. Select your preferred input method (Text or Voice)
        2. Ask your legal question
        3. Receive AI-generated guidance
        4. Consult with a legal professional for specific advice
        
        > **Note:** This is an AI assistant and not a substitute for professional legal advice.
        """)
    
    # Text Input Page
    elif page == "Text Input":
        st.markdown("## 📝 Text Input")
        st.markdown("Enter your legal question below:")
        
        user_input = st.text_area("Your question", height=150,
                                placeholder="E.g., What should I do if my landlord won't return my security deposit?")
        
        if st.button("Get Legal Advice"):
            if user_input.strip():
                with st.spinner("Analyzing your question..."):
                    try:
                        model = load_ai_model()
                        result = get_legal_advice(user_input, model)
                        
                        st.success("Analysis Complete!")
                        
                        st.markdown("### Legal Analysis")
                        
                        # Show query type (punishment or general)
                        query_type = result.get('query_type', 'general')
                        if query_type == 'punishment':
                            st.markdown("🔍 Analyzing your query for relevant legal punishments...")
                        else:
                            st.markdown("🔍 Analyzing your legal query...")
                        
                        st.markdown("### Relevant BNS Sections")
        
                        if not result['relevant_sections']:
                            st.warning("No relevant BNS sections found for your query. Please try rephrasing or consult a legal expert.")
                        else:
                            for section in result['relevant_sections']:
                                with st.expander(f"BNS Section {section['section_number']}: {section['section_title']} (Relevance: {section['score']*100:.1f}%)"):
                                    st.markdown(f"**Description:** {section['description']}")
                                    if 'punishment' in section and section['punishment'] and section['punishment'] != 'Not specified':
                                        st.markdown("---")
                                        st.markdown("#### 🚨 Prescribed Punishment")
                                        st.markdown(f"{section['punishment']}")
                                        st.markdown("""
                                        <style>
                                        .punishment-box {
                                            background-color: #fff8f8;
                                            border-left: 4px solid #ff4b4b;
                                            padding: 0.5rem 1rem;
                                            margin: 1rem 0;
                                            border-radius: 0 4px 4px 0;
                                        }
                                        </style>
                                        <div class='punishment-box'>
                                            <strong>Important:</strong> Punishments may vary based on circumstances, 
                                            prior offenses, and judicial discretion. Always consult with a qualified 
                                            legal professional for case-specific advice.
                                        </div>
                                        """, unsafe_allow_html=True)
                        
                        st.markdown("\n### Suggested Next Steps")
                        for action in result['suggested_actions']:
                            st.markdown(f"- {action}")
                            
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
            else:
                st.warning("Please enter a question before submitting.")
    
    # Voice Input Page
    elif page == "Voice Input":
        st.markdown("## 🎙️ Voice Input")
        st.markdown("Click the button below and speak your legal question:")
        
        if st.button("Start Recording"):
            with st.spinner("Listening... Please speak now."):
                recognizer = VoiceRecognizer()
                success, result = recognizer.listen(timeout=10)
                
                if success:
                    st.session_state.voice_text = result
                    st.success("Speech recognized!")
                    st.text_area("Recognized Text", value=result, height=100)
                    
                    # Process the recognized text
                    with st.spinner("Analyzing your question..."):
                        try:
                            model = load_ai_model()
                            analysis = get_legal_advice(result, model)
                            
                            st.markdown("### Legal Analysis")
                            st.markdown(f"**Category:** {analysis['predicted_category']}")
                            
                            st.markdown("### Suggested Actions")
                            for action in analysis['suggested_actions']:
                                st.markdown(f"- {action}")
                                
                        except Exception as e:
                            st.error(f"An error occurred during analysis: {str(e)}")
                else:
                    st.error(f"Error: {result}")
    
    # About Page
    elif page == "About":
        st.markdown("## About Legal Assistant AI")
        st.markdown("""
        ### Overview
        Legal Assistant AI is designed to provide preliminary legal information and guidance.
        It uses natural language processing to understand legal questions and provide relevant information.
        
        ### Disclaimer
        This application is for informational purposes only and does not constitute legal advice.
        Always consult with a qualified attorney for legal advice specific to your situation.
        
        ### Technical Details
        - Built with Python and Streamlit
        - Uses scikit-learn for text classification
        - Speech recognition for voice input
        
        ### Version
        1.0.0
        """)

if __name__ == "__main__":
    main()
