import streamlit as st
import os
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from model.inference import get_legal_advice, load_model
from speech.voice_to_text import VoiceRecognizer

def display_analysis_results(result, is_image_analysis=False):
    """Helper function to display analysis results concisely and effectively"""
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .section-box {
        border-left: 4px solid #4a90e2;
        padding: 0.75rem 1rem;
        margin: 1rem 0;
        background-color: #f8f9fa;
        border-radius: 0 4px 4px 0;
    }
    .punishment-box {
        border-left: 4px solid #e74c3c;
        padding: 0.75rem 1rem;
        margin: 1rem 0;
        background-color: #fff5f5;
        border-radius: 0 4px 4px 0;
    }
    .section-header {
        color: #2c3e50;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .punishment-header {
        color: #c0392b;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 1rem 0 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # For image analysis, show the scene description first
    if is_image_analysis:
        if 'scene_description' in result:
            st.markdown("### 🖼️ Scene Analysis")
            st.markdown(f"<div class='section-box'>{result['scene_description']}</div>", unsafe_allow_html=True)
        
        # Show detected objects if any
        if 'detected_objects' in result and result['detected_objects']:
            st.markdown("### 🔍 Detected Objects")
            objects_html = ", ".join(
                [f"<span style='font-weight:bold'>{obj['label']}</span> ({obj['score']*100:.0f}%)" 
                 for obj in result['detected_objects']]
            )
            st.markdown(f"<div class='section-box'>{objects_html}</div>", unsafe_allow_html=True)
            
        # Show detected text if any
        if 'detected_text' in result and result['detected_text']:
            st.markdown("### 📝 Detected Text")
            text_html = " | ".join([f"<span style='background-color:#f0f0f0; padding:2px 5px; border-radius:3px;'>{text}</span>" 
                                 for text in result['detected_text']])
            st.markdown(f"<div class='section-box'>{text_html}</div>", unsafe_allow_html=True)
    
    # Show relevant BNS sections
    sections = result.get('sections', []) if is_image_analysis else result.get('relevant_sections', [])
    
    if not sections:
        st.warning("No relevant BNS sections found. Please try rephrasing or consult a legal expert.")
        return
    
    st.markdown("## ⚖️ Relevant BNS Sections")
    
    for section in sections:
        section_title = f"BNS Section {section.get('section_number', 'N/A')}"
        if 'section_title' in section and section['section_title'] and section['section_title'] != 'Untitled':
            section_title += f": {section['section_title']}"
        
        with st.expander(section_title, expanded=True):
            # Show section description with better formatting
            desc = section.get('description', '').strip()
            if desc:
                st.markdown("<div class='section-header'>📜 Section Description</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='section-box'>{desc}</div>", unsafe_allow_html=True)
            
            # Show punishment if available
            if section.get('punishment') and section['punishment'] != 'Not specified':
                punishment = section['punishment'].strip()
                st.markdown("<div class='punishment-header'>🚨 Prescribed Punishment</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='punishment-box'>{punishment}</div>", unsafe_allow_html=True)
                
                # Add a small disclaimer
                st.markdown("""
                <div style='font-size:0.85rem; color:#666; margin-top:0.5rem;'>
                    <i>Note: Punishments may vary based on circumstances and judicial discretion. 
                    Consult a legal professional for specific advice.</i>
                </div>
                """, unsafe_allow_html=True)
    
    # Show suggested actions if any
    if 'suggested_actions' in result and result['suggested_actions']:
        st.markdown("\n**Suggested Next Steps:**")
        for action in result['suggested_actions']:
            st.markdown(f"- {action}")

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
    page = st.sidebar.radio("Go to", ["Home", "Text Input", "Voice Input", "Image Input", "About"])
    
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
        - 📸 Image analysis for legal context
        - ⚡ Quick legal advice
        - 📚 Resource suggestions
        
        ### How to use:
        1. Select your preferred input method (Text, Voice, or Image)
        2. Enter your query or upload an image
        3. Receive AI-generated legal analysis
        4. Review relevant BNS sections and punishments
        5. Consult with a legal professional for specific advice
        
        > **Note:** This is an AI assistant and not a substitute for professional legal advice.
        """)
    
    # Text Input Page
    elif page == "Text Input":
        # Clear voice-related states when on text input page
        if 'voice_text' in st.session_state:
            del st.session_state.voice_text
        if 'voice_processed' in st.session_state:
            del st.session_state.voice_processed
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
        
        # Show recognized text and results if available
        if 'voice_text' in st.session_state and st.session_state.voice_text and st.session_state.voice_processed:
            st.markdown("### Your Question")
            st.info(f'"{st.session_state.voice_text}"')
            
            # Only process if we haven't already processed this exact text
            if 'last_processed_text' not in st.session_state or st.session_state.last_processed_text != st.session_state.voice_text:
                with st.spinner("Analyzing your question..."):
                    try:
                        model = load_ai_model()
                        st.session_state.analysis_result = get_legal_advice(st.session_state.voice_text, model)
                        st.session_state.last_processed_text = st.session_state.voice_text
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                        return
            
            # Show analysis results if available
            if 'analysis_result' in st.session_state and st.session_state.analysis_result:
                display_analysis_results(st.session_state.analysis_result)
                
                # Reset the processed flag to prevent reprocessing
                st.session_state.voice_processed = False
                
            # Show analysis results if available
            if 'analysis_result' in st.session_state and st.session_state.analysis_result:
                result = st.session_state.analysis_result
                
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
            return
            
        # Show recording button if no voice input yet
        st.markdown("Click the button below and speak your legal question:")
        
        if st.button("Start Recording"):
            with st.spinner("Listening... Please speak now."):
                recognizer = VoiceRecognizer()
                success, result = recognizer.listen(timeout=10)
                
                if success:
                    st.session_state.voice_text = result
                    # Process the voice input immediately instead of rerunning
                    st.session_state.user_input = result
                    st.session_state.voice_processed = True
                    st.experimental_rerun()  # Use experimental_rerun for older Streamlit versions
                else:
                    st.error(f"Error: {result}")
                    st.session_state.voice_text = ""
                    st.session_state.user_input = ""
                    st.session_state.voice_processed = False
    
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

    # Image Input Page
    elif page == "Image Input":
        st.markdown("## 📸 Image Analysis")
        st.markdown("Upload an image to analyze its legal context:")
        
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Display the uploaded image
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Analyze Image"):
                with st.spinner("Analyzing image..."):
                    try:
                        # Get the model instance
                        model = load_ai_model()
                        
                        # Check if image processing is available
                        if not hasattr(model, 'analyze_image'):
                            st.error("Image analysis is not available in the current configuration.")
                            st.warning("Please ensure all image processing dependencies are installed.")
                        else:
                            # Analyze the image
                            result = model.analyze_image(uploaded_file)
                            
                            if 'error' in result:
                                st.error(f"Error: {result['error']}")
                            else:
                                st.success("Analysis Complete!")
                                
                                # Display scene description
                                st.markdown("### 🖼️ Scene Analysis")
                                st.info(result.get('scene_description', 'No description available.'))
                                
                                # Display detected objects if any
                                if 'detected_objects' in result and result['detected_objects']:
                                    st.markdown("### 🔍 Detected Objects")
                                    cols = st.columns(3)
                                    for i, obj in enumerate(result['detected_objects']):
                                        with cols[i % 3]:
                                            st.metric(
                                                label=obj.get('label', 'Unknown').title(),
                                                value=f"{obj.get('score', 0)*100:.1f}%"
                                            )
                                
                                # Display detected text if any
                                if 'detected_text' in result and result['detected_text']:
                                    st.markdown("### 📝 Detected Text")
                                    st.code("\n".join(result['detected_text']))
                                
                                # Display relevant BNS sections
                                if 'sections' in result and result['sections']:
                                    st.markdown("### ⚖️ Relevant BNS Sections")
                                    
                                    for section in result['sections']:
                                        with st.expander(f"BNS Section {section.get('section_number', 'N/A')}: {section.get('section_title', 'Untitled')} (Relevance: {section.get('score', 0)*100:.1f}%)"):
                                            st.markdown(f"**Description:** {section.get('description', 'No description available.')}")
                                            
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
                                    
                                    st.markdown("\n### 📋 Suggested Next Steps")
                                    st.markdown("""
                                    - Consult with a legal professional for case-specific advice
                                    - Document all relevant details and evidence
                                    - Review the full text of relevant BNS sections
                                    - Consider filing a report with the appropriate authorities if a crime is suspected
                                    """)
                    
                    except Exception as e:
                        st.error(f"An error occurred during image analysis: {str(e)}")
                        st.exception(e)  # Show full exception for debugging

if __name__ == "__main__":
    main()
