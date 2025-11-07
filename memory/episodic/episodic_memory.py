"""
Episodic Memory Module

Implements the episodic memory system for VANTEX_AI, responsible for storing and retrieving
personal experiences and events with temporal and contextual information.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
import uuid
import numpy as np

from ...utils.vector_store import VectorStore  # Will implement this later

class MemoryConsolidationLevel(Enum):
    """Levels of memory consolidation."""
    SHORT_TERM = auto()     # Recently acquired, high detail
    MEDIUM_TERM = auto()    # Partially consolidated
    LONG_TERM = auto()      # Fully consolidated, abstracted knowledge

@dataclass
class MemoryMetadata:
    """Metadata associated with a memory."""
    timestamp: float
    location: Optional[Dict[str, float]] = None  # GPS coordinates or semantic location
    emotional_valence: Optional[float] = None    # -1.0 (negative) to 1.0 (positive)
    importance: float = 0.5                      # 0.0 to 1.0
    source: str = "system"                       # Source of the memory
    tags: Set[str] = field(default_factory=set)  # For categorization and retrieval

@dataclass
class EpisodicMemory:
    """Represents a single episodic memory."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: Any = None                         # The actual memory content
    embedding: Optional[np.ndarray] = None      # Vector representation for similarity search
    metadata: MemoryMetadata = field(default_factory=MemoryMetadata)
    consolidation: MemoryConsolidationLevel = MemoryConsolidationLevel.SHORT_TERM
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 1
    related_memories: Set[str] = field(default_factory=set)  # IDs of related memories
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to a dictionary for serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "metadata": {
                "timestamp": self.metadata.timestamp,
                "location": self.metadata.location,
                "emotional_valence": self.metadata.emotional_valence,
                "importance": self.metadata.importance,
                "source": self.metadata.source,
                "tags": list(self.metadata.tags)
            },
            "consolidation": self.consolidation.name,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "related_memories": list(self.related_memories)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EpisodicMemory':
        """Create a memory from a dictionary."""
        memory = cls()
        memory.id = data["id"]
        memory.content = data["content"]
        memory.embedding = np.array(data["embedding"]) if data["embedding"] else None
        
        metadata = data.get("metadata", {})
        memory.metadata = MemoryMetadata(
            timestamp=metadata.get("timestamp", time.time()),
            location=metadata.get("location"),
            emotional_valence=metadata.get("emotional_valence"),
            importance=metadata.get("importance", 0.5),
            source=metadata.get("source", "system"),
            tags=set(metadata.get("tags", []))
        )
        
        memory.consolidation = MemoryConsolidationLevel[data.get("consolidation", "SHORT_TERM")]
        memory.last_accessed = data.get("last_accessed", time.time())
        memory.access_count = data.get("access_count", 1)
        memory.related_memories = set(data.get("related_memories", []))
        
        return memory

class EpisodicMemoryStore:
    """Manages storage and retrieval of episodic memories."""
    
    def __init__(self, persist_path: Optional[str] = None):
        """Initialize the episodic memory store.
        
        Args:
            persist_path: Path to persist memories (None for in-memory only)
        """
        self.memories: Dict[str, EpisodicMemory] = {}
        self.vector_store = VectorStore(dimensions=384)  # Using a standard embedding size
        self.persist_path = persist_path
        self._lock = asyncio.Lock()
        
        if self.persist_path:
            self._load_memories()
    
    async def add_memory(
        self,
        content: Any,
        embedding: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """Add a new episodic memory.
        
        Args:
            content: The content to remember (text, image, etc.)
            embedding: Optional vector representation of the content
            metadata: Optional metadata dictionary
            tags: Optional list of tags for categorization
            
        Returns:
            str: ID of the created memory
        """
        if metadata is None:
            metadata = {}
        
        # Create memory object
        memory = EpisodicMemory(
            content=content,
            embedding=embedding,
            metadata=MemoryMetadata(
                timestamp=time.time(),
                location=metadata.get("location"),
                emotional_valence=metadata.get("emotional_valence"),
                importance=metadata.get("importance", 0.5),
                source=metadata.get("source", "system"),
                tags=set(tags or [])
            )
        )
        
        # Add to storage
        async with self._lock:
            self.memories[memory.id] = memory
            
            # Add to vector store if embedding is available
            if memory.embedding is not None:
                self.vector_store.add(memory.id, memory.embedding)
            
            # Persist if configured
            if self.persist_path:
                self._save_memories()
        
        return memory.id
    
    async def get_memory(self, memory_id: str) -> Optional[EpisodicMemory]:
        """Retrieve a memory by its ID."""
        memory = self.memories.get(memory_id)
        if memory:
            # Update access information
            memory.last_accessed = time.time()
            memory.access_count += 1
            
            # Promote memory consolidation level based on access pattern
            self._update_consolidation(memory)
            
            # Persist changes
            if self.persist_path:
                self._save_memories()
                
        return memory
    
    async def search_memories(
        self,
        query_embedding: np.ndarray,
        limit: int = 5,
        min_similarity: float = 0.7,
        tags: Optional[List[str]] = None,
        time_range: Optional[Tuple[float, float]] = None
    ) -> List[Tuple[EpisodicMemory, float]]:
        """Search for similar memories using vector similarity.
        
        Args:
            query_embedding: Query vector for similarity search
            limit: Maximum number of results to return
            min_similarity: Minimum similarity score (0.0 to 1.0)
            tags: Optional list of tags to filter by
            time_range: Optional tuple of (start_time, end_time) to filter by
            
        Returns:
            List of (memory, similarity_score) tuples, sorted by similarity
        """
        # Get similar memory IDs from vector store
        similar_ids = self.vector_store.search(query_embedding, top_k=limit * 2)
        
        # Filter and sort results
        results = []
        for memory_id, similarity in similar_ids:
            if similarity < min_similarity:
                continue
                
            memory = self.memories.get(memory_id)
            if not memory:
                continue
                
            # Apply filters
            if tags and not memory.metadata.tags.intersection(tags):
                continue
                
            if time_range:
                start_time, end_time = time_range
                if not (start_time <= memory.metadata.timestamp <= end_time):
                    continue
            
            results.append((memory, similarity))
            
            # Early termination if we have enough results
            if len(results) >= limit:
                break
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Update access information for retrieved memories
        for memory, _ in results:
            memory.last_accessed = time.time()
            memory.access_count += 1
            self._update_consolidation(memory)
        
        # Persist changes
        if self.persist_path and results:
            self._save_memories()
        
        return results
    
    async def consolidate_memories(self, target_level: MemoryConsolidationLevel):
        """Consolidate memories to the target level."""
        # Implementation would involve:
        # 1. Identifying memories that need consolidation
        # 2. Applying summarization or abstraction
        # 3. Updating consolidation level
        # 4. Updating vector representations
        pass
    
    def _update_consolidation(self, memory: EpisodicMemory):
        """Update the consolidation level of a memory based on access patterns."""
        # Simple heuristic: promote memories that are accessed frequently
        if memory.access_count > 100 and memory.consolidation == MemoryConsolidationLevel.SHORT_TERM:
            memory.consolidation = MemoryConsolidationLevel.MEDIUM_TERM
        elif memory.access_count > 1000 and memory.consolidation == MemoryConsolidationLevel.MEDIUM_TERM:
            memory.consolidation = MemoryConsolidationLevel.LONG_TERM
    
    def _save_memories(self):
        """Save memories to disk."""
        try:
            with open(self.persist_path, 'w') as f:
                json.dump(
                    [m.to_dict() for m in self.memories.values()],
                    f,
                    default=str
                )
        except Exception as e:
            print(f"Error saving memories: {e}")
    
    def _load_memories(self):
        """Load memories from disk."""
        try:
            with open(self.persist_path, 'r') as f:
                memories_data = json.load(f)
                self.memories = {
                    data["id"]: EpisodicMemory.from_dict(data)
                    for data in memories_data
                }
                
                # Rebuild vector store
                for memory in self.memories.values():
                    if memory.embedding is not None:
                        self.vector_store.add(memory.id, memory.embedding)
                        
        except FileNotFoundError:
            # No saved memories yet
            pass
        except Exception as e:
            print(f"Error loading memories: {e}")

# Example usage
if __name__ == "__main__":
    import numpy as np
    
    # Create a memory store with persistence
    memory_store = EpisodicMemoryStore("episodic_memories.json")
    
    # Example: Add a memory
    sample_embedding = np.random.rand(384)  # Example embedding
    memory_id = asyncio.run(
        memory_store.add_memory(
            content="Learned how to ride a bicycle today!",
            embedding=sample_embedding,
            metadata={
                "emotional_valence": 0.9,
                "importance": 0.8,
                "source": "user_input"
            },
            tags=["learning", "personal"]
        )
    )
    
    # Example: Retrieve a memory
    memory = asyncio.run(memory_store.get_memory(memory_id))
    print(f"Retrieved memory: {memory.content}")
    
    # Example: Search for similar memories
    query_embedding = np.random.rand(384)  # Example query embedding
    similar_memories = asyncio.run(
        memory_store.search_memories(
            query_embedding,
            limit=3,
            tags=["learning"]
        )
    )
    
    for mem, similarity in similar_memories:
        print(f"Similar memory ({similarity:.2f}): {mem.content[:50]}...")
