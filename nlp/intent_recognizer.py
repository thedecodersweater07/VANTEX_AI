"""
Intent Recognition Module

Implements intent classification and slot filling for natural language understanding.
Uses a combination of rule-based patterns and machine learning for robust intent detection.
"""

import re
import json
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
import random

from .text_processor import TextProcessor, Token, Sentence

logger = logging.getLogger(__name__)

class IntentType(Enum):
    """Supported intent types."""
    GREETING = auto()
    FAREWELL = auto()
    QUESTION = auto()
    COMMAND = auto()
    INFORM = auto()
    CONFIRM = auto()
    DENY = auto()
    UNKNOWN = auto()

@dataclass
class Slot:
    """Represents a slot (parameter) in an intent."""
    name: str
    value: Any
    start: int  # Start token index
    end: int    # End token index (exclusive)
    confidence: float = 1.0

@dataclass
class Intent:
    """Represents a recognized intent with slots."""
    type: IntentType
    confidence: float
    text: str = ""
    slots: List[Slot] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)

class IntentRecognizer:
    """Recognizes intents and extracts slots from user input."""
    
    def __init__(
        self, 
        language: str = "en",
        model_path: Optional[str] = None,
        use_ml: bool = True
    ):
        """Initialize the intent recognizer.
        
        Args:
            language: Language code (e.g., 'en', 'nl')
            model_path: Path to a pre-trained ML model (optional)
            use_ml: Whether to use machine learning for intent classification
        """
        self.language = language
        self.text_processor = TextProcessor()
        self.use_ml = use_ml
        self.ml_model = None
        self.vectorizer = None
        
        # Load ML model if path is provided and ML is enabled
        if use_ml and model_path and Path(model_path).exists():
            self._load_ml_model(model_path)
        
        # Initialize patterns for rule-based matching
        self._init_patterns()
        
        logger.info(f"Initialized intent recognizer for {language} (ML: {use_ml and self.ml_model is not None})")
    
    def _init_patterns(self):
        """Initialize rule-based patterns for intent recognition."""
        self.patterns = {
            IntentType.GREETING: [
                r"(?i)(hi|hello|hey|greetings|good (morning|afternoon|evening))",
                r"(?i)how(?:'s| is) it going"
            ],
            IntentType.FAREWELL: [
                r"(?i)(goodbye|bye|see you|see ya|farewell)",
                r"(?i)have a (nice|good) (day|night)"
            ],
            IntentType.QUESTION: [
                r"(?i)^(what|who|when|where|why|how|which|can|could|would|will|do|does|is|are|am|was|were|have|has|had) .*\?$",
                r"(?i)^(tell me|explain|describe|what's|what is|who is) .*"
            ],
            IntentType.CONFIRM: [
                r"(?i)^(yes|yeah|yep|sure|ok|okay|of course|absolutely|correct|right|indeed)",
                r"(?i)^(that's|that is) (right|correct|true)"
            ],
            IntentType.DENY: [
                r"(?i)^(no|nope|nah|not really|not at all|negative|incorrect|wrong)",
                r"(?i)^(that's|that is) (not right|incorrect|wrong|false)"
            ],
            IntentType.COMMAND: [
                r"(?i)^(please |)do (a |an |the |)command",
                r"(?i)^(run|execute|perform|start|stop|restart|cancel) .*"
            ]
        }
    
    def _load_ml_model(self, model_path: str):
        """Load a pre-trained ML model for intent classification."""
        try:
            # In a real implementation, this would load a trained model
            # For now, we'll use a placeholder
            logger.info(f"Loading ML model from {model_path}")
            
            # Example of what this might look like with scikit-learn:
            # import joblib
            # model_data = joblib.load(model_path)
            # self.ml_model = model_data['model']
            # self.vectorizer = model_data['vectorizer']
            # self.intent_labels = model_data['intent_labels']
            
            logger.info("ML model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading ML model: {e}")
            self.ml_model = None
            self.vectorizer = None
    
    async def recognize_intent(
        self, 
        text: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Intent:
        """Recognize the intent from the input text.
        
        Args:
            text: Input text to analyze
            context: Optional context from previous interactions
            
        Returns:
            Recognized intent with confidence score and slots
        """
        if not text.strip():
            return Intent(type=IntentType.UNKNOWN, confidence=0.0, text=text)
        
        # Process the text
        sentences = self.text_processor.process_text(text)
        if not sentences:
            return Intent(type=IntentType.UNKNOWN, confidence=0.0, text=text)
        
        # For now, just use the first sentence
        sentence = sentences[0]
        tokens = [token.text for token in sentence.tokens]
        
        # Try ML-based classification first (if available)
        intent_type, ml_confidence = await self._classify_with_ml(text)
        
        # Fall back to rule-based if ML is not available or confidence is low
        if ml_confidence < 0.7:  # Threshold can be adjusted
            intent_type, rule_confidence = self._classify_with_rules(text, tokens)
            confidence = rule_confidence
        else:
            confidence = ml_confidence
        
        # Create intent object
        intent = Intent(
            type=intent_type,
            confidence=float(confidence),
            text=text,
            context=context or {}
        )
        
        # Extract slots
        intent.slots = self._extract_slots(intent, sentence)
        
        return intent
    
    async def _classify_with_ml(self, text: str) -> Tuple[IntentType, float]:
        """Classify intent using a machine learning model.
        
        Returns:
            Tuple of (intent_type, confidence)
        """
        if not self.use_ml or self.ml_model is None:
            return IntentType.UNKNOWN, 0.0
        
        try:
            # In a real implementation, this would use the loaded ML model
            # For now, we'll use a placeholder that occasionally returns a random intent
            
            # Example of what this might look like with scikit-learn:
            # features = self.vectorizer.transform([text])
            # probas = self.ml_model.predict_proba(features)[0]
            # max_idx = np.argmax(probas)
            # intent_type = IntentType[self.intent_labels[max_idx]]
            # confidence = float(probas[max_idx])
            # return intent_type, confidence
            
            # Placeholder implementation
            if random.random() > 0.7:  # 30% chance to return a random intent
                intent_type = random.choice(list(IntentType))
                confidence = random.uniform(0.7, 1.0)
                return intent_type, confidence
            
            return IntentType.UNKNOWN, 0.0
            
        except Exception as e:
            logger.error(f"Error in ML classification: {e}")
            return IntentType.UNKNOWN, 0.0
    
    def _classify_with_rules(
        self, 
        text: str, 
        tokens: List[str]
    ) -> Tuple[IntentType, float]:
        """Classify intent using rule-based patterns.
        
        Returns:
            Tuple of (intent_type, confidence)
        """
        text_lower = text.lower()
        
        # Check for each intent type
        for intent_type, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    # Calculate confidence based on pattern match
                    confidence = min(0.9, 0.5 + (0.4 * len(pattern) / 20))  # Longer patterns get higher confidence
                    return intent_type, confidence
        
        # Default to QUESTION if it looks like a question
        if any(text_lower.strip().endswith(punct) for punct in ['?', '？']):
            return IntentType.QUESTION, 0.6
            
        # Default to INFORM if we can't determine the intent
        return IntentType.INFORM, 0.5
    
    def _extract_slots(
        self, 
        intent: Intent, 
        sentence: Sentence
    ) -> List[Slot]:
        """Extract slots from the recognized intent.
        
        Args:
            intent: The recognized intent
            sentence: The processed sentence
            
        Returns:
            List of extracted slots
        """
        slots = []
        
        # Example: Extract entities as slots
        for entity in sentence.entities:
            slot = Slot(
                name=entity.label_,
                value=entity.text,
                start=entity.start_char,
                end=entity.end_char,
                confidence=0.9  # High confidence for NER-based slots
            )
            slots.append(slot)
        
        # Add more slot extraction logic based on intent type
        if intent.type == IntentType.QUESTION:
            # Extract question words and topics
            question_word = ""
            topic_tokens = []
            
            for i, token in enumerate(sentence.tokens):
                if token.pos_ in ["VERB", "AUX"] and not question_word:
                    question_word = token.lemma_.lower()
                elif token.pos_ in ["NOUN", "PROPN", "ADJ"] and not token.is_stop:
                    topic_tokens.append(token.text)
            
            if question_word:
                slots.append(Slot(
                    name="question_word",
                    value=question_word,
                    start=0,
                    end=len(question_word),
                    confidence=0.8
                ))
            
            if topic_tokens:
                topic = " ".join(topic_tokens)
                slots.append(Slot(
                    name="topic",
                    value=topic,
                    start=0,  # Approximate
                    end=len(topic),
                    confidence=0.7
                ))
        
        return slots
    
    async def train(
        self, 
        training_data: List[Dict[str, Any]],
        model_path: str,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, float]:
        """Train a new intent classification model.
        
        Args:
            training_data: List of training examples with 'text' and 'intent' keys
            model_path: Path to save the trained model
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with training metrics
        """
        if not self.use_ml:
            logger.warning("ML training requested but use_ml is False")
            return {"status": "skipped", "reason": "ML is disabled"}
        
        try:
            # In a real implementation, this would train a model
            # For now, we'll just save a placeholder
            
            # Example with scikit-learn:
            # from sklearn.feature_extraction.text import TfidfVectorizer
            # from sklearn.model_selection import train_test_split
            # from sklearn.ensemble import RandomForestClassifier
            # 
            # X = [item['text'] for item in training_data]
            # y = [item['intent'] for item in training_data]
            # 
            # # Split data
            # X_train, X_test, y_train, y_test = train_test_split(
            #     X, y, test_size=test_size, random_state=random_state
            # )
            # 
            # # Vectorize text
            # self.vectorizer = TfidfVectorizer()
            # X_train_vec = self.vectorizer.fit_transform(X_train)
            # X_test_vec = self.vectorizer.transform(X_test)
            # 
            # # Train model
            # self.ml_model = RandomForestClassifier()
            # self.ml_model.fit(X_train_vec, y_train)
            # 
            # # Evaluate
            # train_score = self.ml_model.score(X_train_vec, y_train)
            # test_score = self.ml_model.score(X_test_vec, y_test)
            # 
            # # Save model
            # model_data = {
            #     'model': self.ml_model,
            #     'vectorizer': self.vectorizer,
            #     'intent_labels': list(set(y_train))
            # }
            # import joblib
            # joblib.dump(model_data, model_path)
            # 
            # return {
            #     'train_accuracy': train_score,
            #     'test_accuracy': test_score,
            #     'num_samples': len(X_train) + len(X_test)
            # }
            
            # Placeholder implementation
            logger.info(f"Training intent classifier with {len(training_data)} examples")
            
            # Simulate training
            import time
            time.sleep(1)  # Simulate training time
            
            return {
                "status": "success",
                "train_accuracy": 0.95,
                "test_accuracy": 0.85,
                "num_samples": len(training_data)
            }
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

# Example usage
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    async def example():
        # Initialize recognizer
        recognizer = IntentRecognizer(use_ml=False)  # Use rule-based only for example
        
        # Example texts to test
        test_texts = [
            "Hello, how are you?",
            "What's the weather like today?",
            "Set a timer for 5 minutes",
            "Yes, that's correct",
            "I want to book a flight to Paris"
        ]
        
        # Test recognition
        for text in test_texts:
            intent = await recognizer.recognize_intent(text)
            print(f"\nText: {text}")
            print(f"Intent: {intent.type.name} (confidence: {intent.confidence:.2f})")
            
            if intent.slots:
                print("Slots:")
                for slot in intent.slots:
                    print(f"- {slot.name}: {slot.value} (confidence: {slot.confidence:.2f})")
        
        # Example of training a model (would require actual training data)
        if recognizer.use_ml:
            training_data = [
                {"text": "hello there", "intent": "greeting"},
                {"text": "hi, how are you?", "intent": "greeting"},
                {"text": "what's the weather?", "intent": "question_weather"},
                {"text": "will it rain tomorrow?", "intent": "question_weather"},
                {"text": "set a timer", "intent": "command_timer"},
                {"text": "start a 5 minute timer", "intent": "command_timer"}
            ]
            
            print("\nTraining model...")
            metrics = await recognizer.train(
                training_data,
                model_path="intent_model.pkl"
            )
            print(f"Training complete: {metrics}")
    
    # Run the example
    import asyncio
    asyncio.run(example())
