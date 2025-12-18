import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Tuple
import sys
import os

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import DETECTOR_MODEL_NAME, DETECTOR_MODEL_PATH


class FakeNewsDetector:
    """BERT-based fake news classifier"""

    def __init__(self, model_name: str = None, model_path: str = None):
        """
        Initialize the fake news detector.

        Args:
            model_name (str): Name of the pretrained model (default from config)
            model_path (str): Path to the fine-tuned model (default from config)
        """
        self.model_name = model_name or DETECTOR_MODEL_NAME
        self.model_path = model_path or DETECTOR_MODEL_PATH
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading tokenizer: {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        print(f"Loading model from: {self.model_path}...")
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        print("FakeNewsDetector initialized successfully.")

    def classify(self, text: str) -> Tuple[str, float]:
        """
        Classify news article as fake or real.

        Args:
            text (str): The news article text

        Returns:
            Tuple[str, float]: (label, confidence) where label is "FAKE" or "REAL"
        """
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**encoding)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][prediction].item()

        # Assuming label 0 = FAKE, label 1 = REAL
        label = "FAKE" if prediction == 0 else "REAL"

        return label, confidence


if __name__ == "__main__":
    # Test the classifier
    try:
        detector = FakeNewsDetector()
        test_text = "This is a sample news article for testing purposes."
        label, confidence = detector.classify(test_text)
        print(f"\nClassification: {label}")
        print(f"Confidence: {confidence:.2%}")
    except Exception as e:
        print(f"Error: {e}")
