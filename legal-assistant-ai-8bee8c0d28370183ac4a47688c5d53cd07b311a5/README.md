# Legal Assistant AI

An AI-powered legal assistant that provides preliminary legal information and guidance through text and voice input.

## Features

- 📝 Text-based legal queries
- 🎙️ Voice input support
- ⚡ Quick legal advice generation
- 📚 Resource suggestions
- 🏷️ Legal document classification

## Prerequisites

- Python 3.8+
- pip (Python package manager)
- Microphone (for voice input)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/legal-assistant-ai.git
   cd legal-assistant-ai
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # On Windows
   # OR
   source venv/bin/activate  # On macOS/Linux
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the required NLTK data and spaCy model:
   ```bash
   python -m nltk.downloader punkt stopwords wordnet
   python -m spacy download en_core_web_sm
   ```

## Usage

### Running the Web Application

To start the Streamlit web interface:

```bash
streamlit run app/main.py
```

Then open your web browser and navigate to `http://localhost:8501`

### Training the Model

To train a new model:

1. Place your training data in `data/raw/` as `legal_cases.csv` with 'text' and 'label' columns
2. Run the training script:
   ```bash
   python model/train_model.py
   ```

### Using Voice Input

1. Ensure you have a working microphone
2. Click the "Start Recording" button in the Voice Input section
3. Speak your legal question clearly
4. The system will process your speech and provide analysis

## Project Structure

```
legal-assistant-ai/
│
├── data/
│   ├── raw/                      # Raw datasets (CSV/JSON)
│   └── cleaned/                  # Cleaned datasets
│
├── model/
│   ├── train_model.py            # Training the ML model
│   └── inference.py              # Predict based on input
│
├── speech/
│   └── voice_to_text.py          # Voice input handling
│
├── app/
│   └── main.py                   # Streamlit web application
│
├── utils/
│   └── text_cleaning.py          # Text preprocessing utilities
│
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This application is for informational purposes only and does not constitute legal advice. Always consult with a qualified attorney for legal advice specific to your situation.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
