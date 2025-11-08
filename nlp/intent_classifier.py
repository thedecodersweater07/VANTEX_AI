"""
Intent Classifier Module

Implements machine learning-based intent classification using scikit-learn.
Supports training, prediction, and model persistence.
"""

import os
import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from .text_processor import TextProcessor

logger = logging.getLogger(__name__)

class IntentType(Enum):
    """Supported intent types."""
    GREETING = "greeting"
    GOODBYE = "goodbye"
    AFFIRM = "affirm"
    DENY = "deny"
    INFORM = "inform"
    REQUEST = "request"
    THANKS = "thanks"
    UNKNOWN = "unknown"

@dataclass
class TrainingExample:
    """A single training example for intent classification."""
    text: str
    intent: str
    language: str = "en"
    entities: List[Dict] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

class IntentClassifier:
    """Machine learning-based intent classifier."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        language: str = "en"
    ):
        """Initialize the intent classifier.
        
        Args:
            model_path: Path to a pre-trained model (optional)
            language: Default language for text processing
        """
        self.language = language
        self.text_processor = TextProcessor()
        self.model = None
        self.vectorizer = None
        self.classes_ = None
        self.language = language
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)
        else:
            self._init_model()
    
    def _init_model(self):
        """Initialize a new model with default parameters."""
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words='english'  # Will be overridden per language
        )
        
        self.model = Pipeline([
            ('tfidf', self.vectorizer),
            ('clf', LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42
            ))
        ])
    
    def preprocess_text(self, text: str, language: str = None) -> str:
        """Preprocess text before classification."""
        lang = language or self.language
        processed = self.text_processor.process_text(text, language=lang)
        if not processed:
            return ""
        
        # Use the first sentence and get lemmatized tokens
        tokens = [token.lemma_.lower() for token in processed[0].tokens 
                 if not token.is_punct and not token.is_space]
        
        return " ".join(tokens)
    
    def train(
        self,
        examples: List[TrainingExample],
        test_size: float = 0.2,
        random_state: int = 42,
        output_dir: str = None
    ) -> Dict[str, Any]:
        """Train the intent classifier.
        
        Args:
            examples: List of training examples
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            output_dir: Directory to save the trained model
            
        Returns:
            Dictionary with training metrics
        """
        if not examples:
            raise ValueError("No training examples provided")
        
        # Preprocess texts and extract labels
        texts = []
        labels = []
        languages = set()
        
        for example in examples:
            processed = self.preprocess_text(example.text, example.language)
            if processed:
                texts.append(processed)
                labels.append(example.intent)
                languages.add(example.language or self.language)
        
        if not texts:
            raise ValueError("No valid training examples after preprocessing")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.classes_ = self.model.named_steps['clf'].classes_.tolist()
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        # Generate classification report
        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        metrics = {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'num_classes': len(self.classes_),
            'num_samples': len(texts),
            'languages': list(languages),
            'classification_report': report
        }
        
        # Save the model if output directory is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            model_path = os.path.join(output_dir, 'intent_classifier.joblib')
            self.save(model_path)
            metrics['model_path'] = model_path
        
        return metrics
    
    def predict(self, text: str, top_n: int = 1) -> List[Dict[str, Any]]:
        """Predict intents for the given text.
        
        Args:
            text: Input text to classify
            top_n: Number of top predictions to return
            
        Returns:
            List of dicts with 'intent' and 'confidence' keys
        """
        if not self.model:
            raise RuntimeError("Model not trained or loaded")
        
        processed = self.preprocess_text(text)
        if not processed:
            return [{"intent": IntentType.UNKNOWN.value, "confidence": 1.0}]
        
        # Get probabilities for all classes
        probas = self.model.predict_proba([processed])[0]
        classes = self.model.named_steps['clf'].classes_
        
        # Sort by probability (descending)
        top_indices = np.argsort(probas)[::-1][:top_n]
        
        results = []
        for idx in top_indices:
            results.append({
                'intent': classes[idx],
                'confidence': float(probas[idx])
            })
        
        return results
    
    def save(self, filepath: str):
        """Save the model to disk."""
        if not self.model:
            raise RuntimeError("No model to save")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the model and metadata
        model_data = {
            'model': self.model,
            'classes': self.classes_ if hasattr(self, 'classes_') else [],
            'language': self.language,
            'version': '1.0.0'
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load a model from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.classes_ = model_data.get('classes', [])
            self.language = model_data.get('language', self.language)
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    @classmethod
    def create_training_data(cls, intents_file: str) -> List[TrainingExample]:
        """Create training data from an intents JSON file.
        
        Expected format:
        {
            "intents": [
                {
                    "tag": "greeting",
                    "patterns": ["Hi", "Hello", "Hey"],
                    "responses": ["Hello!", "Hi there!"],
                    "language": "en"
                },
                ...
            ]
        }
        """
        with open(intents_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        examples = []
        for intent in data.get('intents', []):
            tag = intent.get('tag')
            patterns = intent.get('patterns', [])
            language = intent.get('language', 'en')
            
            for pattern in patterns:
                examples.append(TrainingExample(
                    text=pattern,
                    intent=tag,
                    language=language
                ))
        
        return examples

# Example usage
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Example training data
    training_examples = [
        TrainingExample("Hello!", "greeting"),
        TrainingExample("Hi there", "greeting"),
        TrainingExample("Goodbye", "goodbye"),
        TrainingExample("See you later", "goodbye"),
        TrainingExample("Yes, that's correct", "affirm"),
        TrainingExample("No, that's not right", "deny"),
        TrainingExample("I need help", "request"),
        TrainingExample("Can you help me?", "request"),
        TrainingExample("Thanks!", "thanks"),
        TrainingExample("Thank you very much", "thanks"),
        TrainingExample("My name is John", "inform"),
        TrainingExample("I live in New York", "inform")
    ]
    
    # Initialize and train the classifier
    classifier = IntentClassifier()
    metrics = classifier.train(training_examples, test_size=0.2)
    
    print(f"\nTraining complete!")
    print(f"Train accuracy: {metrics['train_accuracy']:.2f}")
    print(f"Test accuracy: {metrics['test_accuracy']:.2f}")
    
    # Test the classifier
    test_phrases = [
        "Hello!",
        "Goodbye for now",
        "Yes, that's what I meant",
        "No way!",
        "Can you help me with something?",
        "Thanks a lot!",
        "My email is example@test.com"
    ]
    
    print("\nTesting the classifier:")
    for phrase in test_phrases:
        prediction = classifier.predict(phrase)[0]
        print(f"\nText: {phrase}")
        print(f"Predicted: {prediction['intent']} (confidence: {prediction['confidence']:.2f})")
    
    # Save the model
    output_dir = "models/intent"
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "intent_classifier.joblib")
    classifier.save(model_path)
    print(f"\nModel saved to {model_path}")
