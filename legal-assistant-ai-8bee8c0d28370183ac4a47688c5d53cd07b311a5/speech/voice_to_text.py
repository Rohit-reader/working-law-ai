import speech_recognition as sr
import os
from typing import Optional, Tuple

class VoiceRecognizer:
    def __init__(self, energy_threshold=4000, pause_threshold=0.8):
        """Initialize the voice recognizer with custom settings."""
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = energy_threshold
        self.recognizer.pause_threshold = pause_threshold
    
    def listen(self, timeout: int = 5, phrase_time_limit: int = 10) -> Tuple[bool, str]:
        """
        Listen to microphone input and convert speech to text.
        
        Args:
            timeout: Time in seconds to wait for speech before timing out
            phrase_time_limit: Maximum time in seconds for a phrase to be recorded
            
        Returns:
            Tuple of (success: bool, text: str)
        """
        with sr.Microphone() as source:
            print("Listening... (Speak now)")
            try:
                audio = self.recognizer.listen(
                    source, 
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit
                )
                
                print("Processing speech...")
                text = self.recognizer.recognize_google(audio)
                return True, text.lower()
                
            except sr.WaitTimeoutError:
                return False, "No speech detected. Please try again."
            except sr.UnknownValueError:
                return False, "Could not understand audio. Please try again."
            except sr.RequestError as e:
                return False, f"Could not request results; {e}"
            except Exception as e:
                return False, f"An error occurred: {str(e)}"

def record_and_transcribe(timeout: int = 5) -> Tuple[bool, str]:
    """
    Simple function to record and transcribe speech.
    
    Args:
        timeout: Time in seconds to wait for speech before timing out
        
    Returns:
        Tuple of (success: bool, text: str)
    """
    recognizer = VoiceRecognizer()
    return recognizer.listen(timeout=timeout)

if __name__ == "__main__":
    print("Legal Assistant - Voice Input Test")
    print("--------------------------------")
    print("Speak your legal question after the prompt...")
    
    success, result = record_and_transcribe(timeout=5)
    
    if success:
        print("\nYou said:")
        print(f"> {result}")
    else:
        print("\nError:")
        print(f"> {result}")
