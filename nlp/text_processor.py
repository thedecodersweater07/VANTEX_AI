"""
Text Processing Pipeline

Implements the core text processing functionality including tokenization, 
lemmatization, part-of-speech tagging, and named entity recognition.
"""

import re
import spacy
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum, auto
import logging

logger = logging.getLogger(__name__)

class Language(Enum):
    """Supported languages for text processing."""
    ENGLISH = "en"
    DUTCH = "nl"
    # Add more languages as needed

@dataclass
class Token:
    """Represents a processed token with linguistic annotations."""
    text: str
    lemma: str
    pos: str  # Part of speech
    tag: str  # Detailed part of speech
    dep: str  # Dependency relation
    is_stop: bool
    is_punct: bool
    is_alpha: bool
    is_digit: bool
    ent_type: str = ""  # Named entity type
    ent_iob: int = 0    # IOB code for named entities
    start_char: int = 0  # Start character offset
    end_char: int = 0    # End character offset

@dataclass
class Entity:
    """Represents a named entity in text."""
    text: str
    label: str
    start_char: int
    end_char: int
    confidence: float = 1.0

@dataclass
class Sentence:
    """Represents a processed sentence with tokens and entities."""
    text: str
    tokens: List[Token] = field(default_factory=list)
    entities: List[Entity] = field(default_factory=list)
    sentiment: float = 0.0  # Sentiment score from -1.0 (negative) to 1.0 (positive)
    start_char: int = 0
    end_char: int = 0

class TextProcessor:
    """Processes text through an NLP pipeline."""
    
    def __init__(self, language: Language = Language.ENGLISH):
        """Initialize the text processor with the specified language."""
        self.language = language
        self.nlp = self._load_language_model()
        
        # Configure pipeline
        self.disable_pipes = []
        
        logger.info(f"Initialized text processor for {language.name} language")
    
    def _load_language_model(self):
        """Load the appropriate language model."""
        model_name = {
            Language.ENGLISH: "en_core_web_md",  # Medium English model
            Language.DUTCH: "nl_core_news_md"    # Medium Dutch model
        }.get(self.language, "xx_ent_wiki_sm")   # Multi-language as fallback
        
        try:
            return spacy.load(model_name)
        except OSError:
            logger.warning(
                f"Model {model_name} not found. Downloading..."
            )
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", model_name])
            return spacy.load(model_name)
    
    def process_text(
        self, 
        text: str, 
        disable: Optional[List[str]] = None
    ) -> List[Sentence]:
        """Process the input text through the NLP pipeline.
        
        Args:
            text: Input text to process
            disable: List of pipeline components to disable
            
        Returns:
            List of processed sentences
        """
        if not text.strip():
            return []
            
        # Process with spaCy
        doc = self.nlp(text, disable=disable or [])
        
        # Convert to our data structures
        sentences = []
        for sent in doc.sents:
            sentence = Sentence(
                text=sent.text,
                start_char=sent.start_char,
                end_char=sent.end_char
            )
            
            # Process tokens
            for token in sent:
                sentence.tokens.append(Token(
                    text=token.text,
                    lemma=token.lemma_, 
                    pos=token.pos_,
                    tag=token.tag_,
                    dep=token.dep_,
                    is_stop=token.is_stop,
                    is_punct=token.is_punct,
                    is_alpha=token.is_alpha,
                    is_digit=token.is_digit,
                    ent_type=token.ent_type_,
                    ent_iob=token.ent_iob_,
                    start_char=token.idx,
                    end_char=token.idx + len(token)
                ))
            
            # Process entities
            for ent in sent.ents:
                sentence.entities.append(Entity(
                    text=ent.text,
                    label=ent.label_,
                    start_char=ent.start_char,
                    end_char=ent.end_char
                ))
            
            # Basic sentiment analysis (placeholder - would use a real model in production)
            sentence.sentiment = self._estimate_sentiment(sentence)
            
            sentences.append(sentence)
        
        return sentences
    
    def _estimate_sentiment(self, sentence: Sentence) -> float:
        """Estimate sentiment of a sentence."""
        # Simple rule-based sentiment as fallback
        # In production, replace with a proper sentiment analysis model
        
        positive_words = {"good", "great", "excellent", "awesome", "happy", "like", "love"}
        negative_words = {"bad", "terrible", "awful", "hate", "dislike", "sad"}
        
        score = 0.0
        for token in sentence.tokens:
            if token.is_alpha and not token.is_stop:
                if token.lemma.lower() in positive_words:
                    score += 0.2
                elif token.lemma.lower() in negative_words:
                    score -= 0.2
        
        # Normalize to [-1, 1] range
        return max(-1.0, min(1.0, score))
    
    def extract_keywords(
        self, 
        text: str, 
        top_n: int = 10,
        include_pos: List[str] = None,
        exclude_pos: List[str] = None
    ) -> List[Tuple[str, float]]:
        """Extract keywords from text using TF-IDF.
        
        Args:
            text: Input text
            top_n: Number of keywords to return
            include_pos: Only include these part-of-speech tags
            exclude_pos: Exclude these part-of-speech tags
            
        Returns:
            List of (keyword, score) tuples, sorted by score
        """
        from collections import defaultdict
        
        # Default POS filters
        if include_pos is None:
            include_pos = ['NOUN', 'PROPN', 'ADJ']
        if exclude_pos is None:
            exclude_pos = []
        
        # Process text
        doc = self.nlp(text)
        
        # Count term frequencies
        term_freq = defaultdict(int)
        for token in doc:
            if (token.pos_ in include_pos and 
                token.pos_ not in exclude_pos and
                not token.is_stop and 
                not token.is_punct and
                token.is_alpha):
                term_freq[token.lemma_.lower()] += 1
        
        # Simple scoring (TF)
        total_terms = sum(term_freq.values())
        if total_terms == 0:
            return []
            
        # Calculate scores
        scores = [
            (term, freq / total_terms) 
            for term, freq in term_freq.items()
        ]
        
        # Sort by score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:top_n]
    
    def get_noun_chunks(self, text: str) -> List[str]:
        """Extract noun chunks from text."""
        doc = self.nlp(text)
        return [chunk.text for chunk in doc.noun_chunks]
    
    def get_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts."""
        doc1 = self.nlp(text1)
        doc2 = self.nlp(text2)
        return doc1.similarity(doc2)

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize processor
    processor = TextProcessor(Language.ENGLISH)
    
    # Example text
    text = """
    VANTEX_AI is an advanced AI system developed by VANTEX. 
    It can understand and process natural language with high accuracy. 
    The system is designed to be modular and extensible.
    """
    
    # Process text
    sentences = processor.process_text(text)
    
    # Print results
    print("Processed Sentences:")
    for i, sent in enumerate(sentences, 1):
        print(f"\nSentence {i}: {sent.text}")
        print(f"Sentiment: {sent.sentiment:.2f}")
        
        print("\nEntities:")
        for ent in sent.entities:
            print(f"- {ent.text} ({ent.label_})")
    
    # Extract keywords
    print("\nKeywords:")
    keywords = processor.extract_keywords(text, top_n=5)
    for kw, score in keywords:
        print(f"- {kw}: {score:.3f}")
    
    # Get noun chunks
    print("\nNoun chunks:")
    for chunk in processor.get_noun_chunks(text):
        print(f"- {chunk}")
    
    # Compare similarity
    text1 = "The cat sat on the mat"
    text2 = "A feline is sitting on a rug"
    similarity = processor.get_similarity(text1, text2)
    print(f"\nSimilarity between texts: {similarity:.2f}")
