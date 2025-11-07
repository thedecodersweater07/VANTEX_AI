"""
Working Memory Module

Implements the working memory system for VANTEX_AI, responsible for maintaining
and manipulating the system's current state, context, and focus of attention.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Set, Deque, Union
from dataclasses import dataclass, field
from collections import deque
from enum import Enum, auto
import uuid

class WorkingMemoryState(Enum):
    """Possible states of working memory."""
    ACTIVE = auto()        # Actively processing information
    MAINTENANCE = auto()   # Consolidating or reorganizing
    IDLE = auto()          # Minimal activity, ready for new input
    OVERLOADED = auto()    # Approaching or at capacity

@dataclass
class MemoryChunk:
    """A chunk of information in working memory."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: Any = None
    content_type: str = "text"  # text, image, audio, etc.
    priority: float = 0.5       # 0.0 (low) to 1.0 (high)
    timestamp: float = field(default_factory=time.time)
    expiration: Optional[float] = None  # When this chunk should expire
    source: str = "system"
    metadata: Dict[str, Any] = field(default_factory=dict)
    related_chunks: Set[str] = field(default_factory=set)  # IDs of related chunks

class WorkingMemory:
    """Manages the system's working memory with capacity limits and decay."""
    
    def __init__(
        self,
        max_chunks: int = 7,  # Magic number 7±2 from cognitive psychology
        decay_rate: float = 0.1,  # Rate at which chunk importance decays over time
        chunk_ttl: Optional[float] = 300.0  # Default time-to-live in seconds
    ):
        """Initialize working memory.
        
        Args:
            max_chunks: Maximum number of chunks to store
            decay_rate: Rate at which chunk importance decays (0.0 to 1.0)
            chunk_ttl: Default time-to-live for chunks in seconds (None for no expiration)
        """
        self.max_chunks = max(1, max_chunks)  # At least 1 chunk
        self.decay_rate = min(1.0, max(0.0, decay_rate))
        self.default_ttl = chunk_ttl
        
        self.chunks: Dict[str, MemoryChunk] = {}
        self.chunk_queue: Deque[str] = deque(maxlen=max_chunks)  # For FIFO eviction
        self.current_state = WorkingMemoryState.IDLE
        self.last_update = time.time()
        self._lock = asyncio.Lock()
    
    async def add_chunk(
        self,
        content: Any,
        content_type: str = "text",
        priority: float = 0.5,
        ttl: Optional[float] = None,
        source: str = "system",
        metadata: Optional[Dict[str, Any]] = None,
        related_chunks: Optional[List[str]] = None
    ) -> str:
        """Add a new chunk to working memory.
        
        Args:
            content: The content to store
            content_type: Type of content (text, image, etc.)
            priority: Importance of this chunk (0.0 to 1.0)
            ttl: Time-to-live in seconds (None for default)
            source: Source of this chunk
            metadata: Additional metadata
            related_chunks: IDs of related memory chunks
            
        Returns:
            str: ID of the created chunk
        """
        # Apply bounds to priority
        priority = max(0.0, min(1.0, priority))
        
        # Create the chunk
        chunk = MemoryChunk(
            content=content,
            content_type=content_type,
            priority=priority,
            source=source,
            metadata=metadata or {},
            related_chunks=set(related_chunks or [])
        )
        
        # Set expiration if TTL is provided
        if ttl is not None or self.default_ttl is not None:
            chunk.expiration = time.time() + (ttl if ttl is not None else self.default_ttl)
        
        async with self._lock:
            # Check if we need to evict chunks
            self._enforce_capacity()
            
            # Add the new chunk
            self.chunks[chunk.id] = chunk
            self.chunk_queue.append(chunk.id)
            
            # Update state
            self._update_state()
            
        return chunk.id
    
    async def get_chunk(self, chunk_id: str) -> Optional[MemoryChunk]:
        """Retrieve a chunk by ID."""
        chunk = self.chunks.get(chunk_id)
        if chunk:
            # Update last access time
            chunk.timestamp = time.time()
            
            # Move to end of queue (most recently used)
            if chunk_id in self.chunk_queue:
                self.chunk_queue.remove(chunk_id)
                self.chunk_queue.append(chunk_id)
            
            # Update state
            self._update_state()
            
        return chunk
    
    async def update_chunk(
        self,
        chunk_id: str,
        content: Any = None,
        priority: Optional[float] = None,
        ttl: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update an existing chunk.
        
        Args:
            chunk_id: ID of the chunk to update
            content: New content (None to keep existing)
            priority: New priority (None to keep existing)
            ttl: New time-to-live in seconds (None to keep existing)
            metadata: Metadata to update (shallow merge with existing)
            
        Returns:
            bool: True if chunk was updated, False if not found
        """
        if chunk_id not in self.chunks:
            return False
            
        chunk = self.chunks[chunk_id]
        
        # Update fields if provided
        if content is not None:
            chunk.content = content
        if priority is not None:
            chunk.priority = max(0.0, min(1.0, priority))
        if ttl is not None:
            chunk.expiration = time.time() + ttl if ttl > 0 else None
        if metadata is not None:
            chunk.metadata.update(metadata)
            
        # Update timestamp
        chunk.timestamp = time.time()
        
        # Move to end of queue (most recently used)
        if chunk_id in self.chunk_queue:
            self.chunk_queue.remove(chunk_id)
            self.chunk_queue.append(chunk_id)
        
        # Update state
        self._update_state()
        
        return True
    
    async def remove_chunk(self, chunk_id: str) -> bool:
        """Remove a chunk from working memory.
        
        Returns:
            bool: True if chunk was removed, False if not found
        """
        if chunk_id not in self.chunks:
            return False
            
        async with self._lock:
            # Remove from chunks dict
            self.chunks.pop(chunk_id, None)
            
            # Remove from queue if present
            if chunk_id in self.chunk_queue:
                self.chunk_queue.remove(chunk_id)
            
            # Remove references from other chunks
            for other_chunk in self.chunks.values():
                if chunk_id in other_chunk.related_chunks:
                    other_chunk.related_chunks.remove(chunk_id)
            
            # Update state
            self._update_state()
            
        return True
    
    async def get_context(self, limit: int = 5) -> List[MemoryChunk]:
        """Get the most relevant chunks as context.
        
        Args:
            limit: Maximum number of chunks to return
            
        Returns:
            List of MemoryChunk objects, sorted by relevance
        """
        self._cleanup_expired()
        
        # Simple relevance scoring based on priority and recency
        current_time = time.time()
        chunks = list(self.chunks.values())
        
        # Calculate scores
        scored_chunks = []
        for chunk in chunks:
            # Base score on priority
            score = chunk.priority
            
            # Apply recency decay
            age = current_time - chunk.timestamp
            decay = 1.0 / (1.0 + self.decay_rate * age)
            score *= decay
            
            scored_chunks.append((score, chunk))
        
        # Sort by score (descending)
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        # Return top N chunks
        return [chunk for _, chunk in scored_chunks[:limit]]
    
    async def link_chunks(self, chunk_id1: str, chunk_id2: str, bidirectional: bool = True) -> bool:
        """Create a relationship between two chunks.
        
        Args:
            chunk_id1: First chunk ID
            chunk_id2: Second chunk ID
            bidirectional: If True, create links in both directions
            
        Returns:
            bool: True if both chunks exist and were linked
        """
        if chunk_id1 not in self.chunks or chunk_id2 not in self.chunks:
            return False
            
        self.chunks[chunk_id1].related_chunks.add(chunk_id2)
        if bidirectional:
            self.chunks[chunk_id2].related_chunks.add(chunk_id1)
            
        return True
    
    async def get_related_chunks(self, chunk_id: str) -> List[MemoryChunk]:
        """Get chunks related to the specified chunk."""
        if chunk_id not in self.chunks:
            return []
            
        related = []
        for related_id in self.chunks[chunk_id].related_chunks:
            if related_id in self.chunks:
                related.append(self.chunks[related_id])
                
        return related
    
    async def clear(self):
        """Clear all chunks from working memory."""
        async with self._lock:
            self.chunks.clear()
            self.chunk_queue.clear()
            self.current_state = WorkingMemoryState.IDLE
            self.last_update = time.time()
    
    def get_state(self) -> WorkingMemoryState:
        """Get the current state of working memory."""
        return self.current_state
    
    def _update_state(self):
        """Update the current state based on memory usage."""
        usage = len(self.chunks) / self.max_chunks if self.max_chunks > 0 else 0.0
        
        if usage >= 0.9:
            self.current_state = WorkingMemoryState.OVERLOADED
        elif usage >= 0.6:
            self.current_state = WorkingMemoryState.ACTIVE
        elif usage > 0.1:
            self.current_state = WorkingMemoryState.MAINTENANCE
        else:
            self.current_state = WorkingMemoryState.IDLE
            
        self.last_update = time.time()
    
    def _enforce_capacity(self):
        """Ensure we don't exceed maximum capacity by evicting chunks if needed."""
        self._cleanup_expired()
        
        # If still over capacity, remove least recently used chunks
        while len(self.chunks) >= self.max_chunks and self.chunk_queue:
            chunk_id = self.chunk_queue.popleft()
            self.chunks.pop(chunk_id, None)
    
    def _cleanup_expired(self):
        """Remove expired chunks."""
        current_time = time.time()
        expired_ids = [
            chunk_id for chunk_id, chunk in self.chunks.items()
            if chunk.expiration is not None and chunk.expiration <= current_time
        ]
        
        for chunk_id in expired_ids:
            self.chunks.pop(chunk_id, None)
            if chunk_id in self.chunk_queue:
                self.chunk_queue.remove(chunk_id)

# Example usage
if __name__ == "__main__":
    async def main():
        # Create working memory with small capacity for testing
        wm = WorkingMemory(max_chunks=3, chunk_ttl=10.0)
        
        # Add some chunks
        chunk1 = await wm.add_chunk(
            content="The user's name is Alice",
            priority=0.8,
            source="user_input",
            metadata={"intent": "introduce"}
        )
        
        chunk2 = await wm.add_chunk(
            content="Current task: Answer questions about AI",
            priority=0.9,
            source="system",
            metadata={"task": "qa", "domain": "AI"}
        )
        
        # Link related chunks
        await wm.link_chunks(chunk1, chunk2)
        
        # Get current context
        context = await wm.get_context()
        print("Current context:")
        for i, chunk in enumerate(context, 1):
            print(f"{i}. {chunk.content} (priority: {chunk.priority:.2f})")
        
        # Update a chunk
        await wm.update_chunk(chunk1, priority=0.95)
        
        # Get updated chunk
        updated = await wm.get_chunk(chunk1)
        print(f"\nUpdated priority: {updated.priority if updated else 'N/A'}")
        
        # Get related chunks
        related = await wm.get_related_chunks(chunk1)
        print("\nRelated chunks:")
        for chunk in related:
            print(f"- {chunk.content}")
        
        # Clear working memory
        await wm.clear()
        print(f"\nAfter clear, chunk count: {len(wm.chunks)}")
    
    asyncio.run(main())
