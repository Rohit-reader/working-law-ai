import streamlit as st
import os
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from model.inference import get_legal_advice, load_model
from speech.voice_to_text import VoiceRecognizer

def display_analysis_results(result, is_image_analysis=False):
    """Helper function to display analysis results with enhanced styling"""
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .section-box {
        border-left: 4px solid #4a90e2;
        padding: 1rem 1.25rem;
        margin: 1.25rem 0;
        background-color: #1e293b;
        border-radius: 0 8px 8px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: white;
    }
    .punishment-box {
        border-left: 4px solid #ef4444;
        padding: 1rem 1.25rem;
        margin: 1.5rem 0;
        background-color: #1e1b4b;
        border-radius: 0 8px 8px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: white;
    }
    .section-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #60a5fa;
        margin-bottom: 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #334155;
    }
    .section-content {
        font-size: 1rem;
        line-height: 1.6;
        color: #e2e8f0;
        margin: 0.5rem 0;
    }
    .punishment-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #f87171;
        margin: 0.5rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .punishment-header:before {
        content: "⚠️";
        font-size: 1.2em;
    }
    .punishment-content {
        font-size: 1rem;
        line-height: 1.6;
        color: #e2e8f0;
        background-color: rgba(239, 68, 68, 0.1);
        padding: 0.75rem;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
    .legal-note {
        font-size: 0.85rem;
        color: #94a3b8;
        margin: 0.75rem 0 0.25rem 0;
        padding: 0.5rem;
        background-color: rgba(30, 41, 59, 0.5);
        border-radius: 4px;
        border-left: 2px solid #475569;
    }
    .suggested-actions {
        margin: 1.5rem 0 0.5rem 0;
        padding: 1rem;
        background-color: #0f172a;
        border-radius: 8px;
        border-left: 4px solid #7c3aed;
    }
    .suggested-actions-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #a78bfa;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .suggested-actions-list {
        padding-left: 1.25rem;
        margin: 0.5rem 0;
    }
    .suggested-actions-list li {
        margin: 0.5rem 0;
        color: #cbd5e1;
    }
    </style>
    """, unsafe_allow_html=True)
    
    if is_image_analysis:
        # Display image analysis results
        st.markdown("<h2 style='color:#60a5fa; margin-bottom:1.5rem;'>📸 Image Analysis Results</h2>", unsafe_allow_html=True)
        
        # Show scene description
        if 'scene_description' in result:
            st.markdown(f"""
            <div class="section-box">
                <div class="section-header">🔍 Scene Analysis</div>
                <div class="section-content">{result['scene_description']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Show detected objects
        if 'detected_objects' in result and result['detected_objects']:
            objects_html = ""
            for obj in result['detected_objects']:
                # Create a visual bar for confidence level
                width = min(100, int(obj['score'] * 1.5))  # Scale for better visibility
                objects_html += f"""
                <div style="margin: 0.5rem 0;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                        <span>{obj['label'].title()}</span>
                        <span style="font-weight: 600;">{obj['score']:.1f}%</span>
                    </div>
                    <div style="height: 6px; background: #334155; border-radius: 3px; overflow: hidden;">
                        <div style="height: 100%; width: {width}%; background: #60a5fa;"></div>
                    </div>
                </div>
                """
            
            st.markdown(f"""
            <div class="section-box">
                <div class="section-header">📋 Detected Legal Concepts</div>
                <div class="section-content">{objects_html}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Show detected text if any
        if 'detected_text' in result and result['detected_text']:
            text_items = "".join([f"<li>{text}</li>" for text in result['detected_text']])
            st.markdown(f"""
            <div class="section-box">
                <div class="section-header">✏️ Detected Text</div>
                <div class="section-content">
                    <ul style="margin: 0.5rem 0 0 1rem; padding-left: 1rem;">
                        {text_items}
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Display relevant BNS sections
    if 'relevant_sections' in result and result['relevant_sections']:
        st.markdown("<h2 style='color:#60a5fa; margin:2rem 0 1.5rem 0;'>⚖️ Relevant BNS Sections</h2>", unsafe_allow_html=True)
        
        for idx, section in enumerate(result['relevant_sections'], 1):
            # Section header and description
            st.markdown(f"""
            <div class="section-box">
                <div class="section-header">
                    <span style="color: #93c5fd;">Section {section.get('section_number', 'N/A')}:</span> 
                    {section.get('section_title', 'No Title')}
                </div>
                <div class="section-content">
                    {section.get('description', 'No description available.')}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Punishment information with enhanced styling
            if 'punishment' in section and section['punishment']:
                st.markdown(f"""
                <div class="punishment-box">
                    <div class="punishment-header">Punishment</div>
                    <div class="punishment-content">
                        {section['punishment']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Legal disclaimer
                st.markdown("""
                <div class="legal-note">
                    <i>ℹ️ Note: Punishments may vary based on circumstances, prior offenses, and judicial discretion. 
                    This is not legal advice. Please consult a qualified legal professional for specific guidance.</i>
                </div>
                """, unsafe_allow_html=True)
    
    # Show suggested actions if any
    if 'suggested_actions' in result and result['suggested_actions']:
        actions_html = "".join([f"<li>{action}</li>" for action in result['suggested_actions']])
        st.markdown(f"""
        <div class="suggested-actions">
            <div class="suggested-actions-title">📌 Recommended Actions</div>
            <ul class="suggested-actions-list">
                {actions_html}
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Add a subtle separator
    st.markdown("<div style='height: 1px; background: linear-gradient(90deg, transparent, #334155, transparent); margin: 2rem 0;'></div>", unsafe_allow_html=True)

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
