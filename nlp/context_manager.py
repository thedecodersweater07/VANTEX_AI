"""
Context Manager Module

Manages conversation context and state across multiple turns of interaction.
"""

import time
import json
import logging
import uuid
from typing import Dict, List, Optional, Any, Deque
from dataclasses import dataclass, field
from collections import deque
from enum import Enum, auto

logger = logging.getLogger(__name__)

class ContextType(Enum):
    """Types of context that can be stored."""
    CONVERSATION = "conversation"
    USER = "user"
    SESSION = "session"

@dataclass
class ContextEntity:
    """Represents a contextual entity with a value and metadata."""
    value: Any
    source: str = "system"
    confidence: float = 1.0
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class ContextManager:
    """Manages conversation context and state."""
    
    def __init__(self, user_id: str = None, session_id: str = None, max_history: int = 20):
        self.user_id = user_id or str(uuid.uuid4())
        self.session_id = session_id or str(uuid.uuid4())
        self.max_history = max_history
        
        # Context storage by type
        self.contexts = {t: {} for t in ContextType}
        self.history = deque(maxlen=max_history)
        
        # Initialize with basic context
        self._init_basic_context()
    
    def _init_basic_context(self):
        """Initialize with basic context."""
        self.set_context("session_id", self.session_id, ContextType.SESSION)
        self.set_context("user_id", self.user_id, ContextType.USER)
        self.set_context("session_start", time.time(), ContextType.SESSION)
    
    def set_context(self, key: str, value: Any, context_type: ContextType = ContextType.CONVERSATION,
                   source: str = "system", ttl: float = None, **metadata):
        """Set a context value."""
        expires_at = time.time() + ttl if ttl else None
        entity = ContextEntity(
            value=value,
            source=source,
            expires_at=expires_at,
            metadata=metadata
        )
        self.contexts[context_type][key] = entity
        return entity
    
    def get_context(self, key: str, context_type: ContextType = None, default=None):
        """Get a context value."""
        if context_type:
            if key in self.contexts[context_type]:
                entity = self.contexts[context_type][key]
                if not entity.expires_at or entity.expires_at > time.time():
                    return entity.value
            return default
        
        # Search all context types
        for ctx_type in [ContextType.CONVERSATION, ContextType.SESSION, ContextType.USER]:
            if key in self.contexts[ctx_type]:
                entity = self.contexts[ctx_type][key]
                if not entity.expires_at or entity.expires_at > time.time():
                    return entity.value
        return default
    
    def add_to_history(self, role: str, content: str, intent: dict = None, **metadata):
        """Add a message to the conversation history."""
        entry = {
            "id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "role": role,
            "content": content,
            "intent": intent,
            "metadata": metadata
        }
        self.history.append(entry)
        return entry
    
    def get_conversation_summary(self, max_turns: int = 5) -> str:
        """Generate a summary of the conversation."""
        if not self.history:
            return "No conversation history."
        
        turns = list(self.history)[-max_turns:]
        return "\n".join(
            f"{turn['role'].capitalize()}: {turn['content']}" 
            for turn in turns
        )

# Example usage
if __name__ == "__main__":
    # Initialize context manager
    cm = ContextManager(user_id="user123")
    
    # Set some context
    cm.set_context("user_name", "Alice", ContextType.USER)
    cm.set_context("current_task", "weather_check", ttl=300)  # 5 min TTL
    
    # Add conversation history
    cm.add_to_history("user", "What's the weather like?")
    cm.add_to_history("assistant", "I'll check the weather for you.")
    
    # Get context
    print(f"User: {cm.get_context('user_name')}")
    print(f"Current task: {cm.get_context('current_task')}")
    
    # Print conversation summary
    print("\nConversation Summary:")
    print(cm.get_conversation_summary())
