"""
Memory Consolidation Module

Implements memory consolidation processes that transfer information between different
memory systems (working -> episodic/semantic) and perform memory maintenance.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
import numpy as np

from .episodic.episodic_memory import EpisodicMemoryStore, MemoryConsolidationLevel
from .semantic.semantic_memory import SemanticMemory, Concept, RelationshipType
from .working.working_memory import WorkingMemory, MemoryChunk

logger = logging.getLogger(__name__)

class ConsolidationStrategy(Enum):
    """Strategies for memory consolidation."""
    TIME_BASED = auto()      # Consolidate based on time intervals
    EVENT_BASED = auto()     # Consolidate after specific events
    LOAD_BASED = auto()      # Consolidate when working memory load is high
    HYBRID = auto()          # Combination of the above strategies

@dataclass
class ConsolidationConfig:
    """Configuration for memory consolidation."""
    strategy: ConsolidationStrategy = ConsolidationStrategy.HYBRID
    interval: float = 300.0  # Seconds between time-based consolidations
    min_chunk_age: float = 60.0  # Minimum age (seconds) for chunks to be consolidated
    importance_threshold: float = 0.7  # Minimum importance for consolidation
    batch_size: int = 5  # Max chunks to process per consolidation cycle
    max_consolidation_time: float = 30.0  # Max seconds to spend on consolidation

class MemoryConsolidator:
    """Manages the consolidation of memories between memory systems."""
    
    def __init__(
        self,
        working_memory: WorkingMemory,
        episodic_memory: Optional[EpisodicMemoryStore] = None,
        semantic_memory: Optional[SemanticMemory] = None,
        config: Optional[ConsolidationConfig] = None
    ):
        """Initialize the memory consolidator.
        
        Args:
            working_memory: The working memory instance
            episodic_memory: The episodic memory store
            semantic_memory: The semantic memory store
            config: Consolidation configuration
        """
        self.working_memory = working_memory
        self.episodic_memory = episodic_memory
        self.semantic_memory = semantic_memory
        self.config = config or ConsolidationConfig()
        
        self._last_consolidation = time.time()
        self._is_consolidating = False
        self._stop_event = asyncio.Event()
        self._consolidation_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the background consolidation process."""
        if self._consolidation_task is not None and not self._consolidation_task.done():
            logger.warning("Consolidation already running")
            return
            
        self._stop_event.clear()
        self._consolidation_task = asyncio.create_task(self._consolidation_loop())
    
    async def stop(self):
        """Stop the background consolidation process."""
        if self._consolidation_task is None:
            return
            
        self._stop_event.set()
        try:
            await asyncio.wait_for(self._consolidation_task, timeout=5.0)
        except asyncio.TimeoutError:
            self._consolidation_task.cancel()
        
        self._consolidation_task = None
    
    async def _consolidation_loop(self):
        """Background loop for periodic consolidation."""
        logger.info("Starting memory consolidation loop")
        
        while not self._stop_event.is_set():
            try:
                # Check if it's time to consolidate
                time_since_last = time.time() - self._last_consolidation
                
                if (
                    self.config.strategy in [ConsolidationStrategy.TIME_BASED, ConsolidationStrategy.HYBRID] and
                    time_since_last >= self.config.interval
                ):
                    await self.consolidate()
                
                # Short sleep to prevent busy waiting
                await asyncio.sleep(1.0)
                
            except asyncio.CancelledError:
                logger.info("Consolidation loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in consolidation loop: {e}", exc_info=True)
                await asyncio.sleep(5.0)  # Prevent tight error loops
    
    async def consolidate(self, force: bool = False) -> bool:
        """Perform memory consolidation.
        
        Args:
            force: If True, force consolidation even if conditions aren't ideal
            
        Returns:
            bool: True if consolidation was performed, False otherwise
        """
        # Prevent concurrent consolidations
        if self._is_consolidating:
            logger.debug("Consolidation already in progress")
            return False
            
        self._is_consolidating = True
        start_time = time.time()
        consolidated = False
        
        try:
            # Check if we should consolidate based on strategy
            should_consolidate = force
            
            if not should_consolidate and self.config.strategy in [ConsolidationStrategy.LOAD_BASED, ConsolidationStrategy.HYBRID]:
                # Check working memory load
                wm_state = self.working_memory.get_state()
                should_consolidate = wm_state == WorkingMemory.WorkingMemoryState.OVERLOADED
                
                if should_consolidate:
                    logger.info("Consolidation triggered by high working memory load")
            
            if not should_consolidate and self.config.strategy == ConsolidationStrategy.TIME_BASED:
                # Time-based consolidation
                time_since_last = time.time() - self._last_consolidation
                should_consolidate = time_since_last >= self.config.interval
                
                if should_consolidate:
                    logger.debug(f"Time-based consolidation (last: {time_since_last:.1f}s ago)")
            
            if not should_consolidate and not force:
                return False
            
            logger.info("Starting memory consolidation...")
            
            # Get chunks eligible for consolidation
            eligible_chunks = await self._get_eligible_chunks()
            
            if not eligible_chunks:
                logger.debug("No chunks eligible for consolidation")
                return False
            
            # Process chunks (up to batch size)
            processed = 0
            for chunk in eligible_chunks[:self.config.batch_size]:
                if time.time() - start_time > self.config.max_consolidation_time:
                    logger.warning(f"Consolidation timeout after {processed} chunks")
                    break
                
                try:
                    consolidated |= await self._consolidate_chunk(chunk)
                    processed += 1
                except Exception as e:
                    logger.error(f"Error consolidating chunk {chunk.id}: {e}", exc_info=True)
            
            logger.info(f"Consolidated {processed} chunks in {time.time() - start_time:.2f}s")
            self._last_consolidation = time.time()
            return consolidated
            
        finally:
            self._is_consolidating = False
    
    async def _get_eligible_chunks(self) -> List[MemoryChunk]:
        """Get chunks from working memory that are eligible for consolidation."""
        current_time = time.time()
        eligible = []
        
        # Get all chunks from working memory
        chunks = list((await self.working_memory.get_context(limit=100)))
        
        for chunk in chunks:
            # Skip chunks that are too new
            chunk_age = current_time - chunk.timestamp
            if chunk_age < self.config.min_chunk_age:
                continue
                
            # Skip low-importance chunks
            if chunk.priority < self.config.importance_threshold:
                continue
                
            # Skip chunks with future expiration
            if chunk.expiration is not None and chunk.expiration > current_time + 60:  # 1 min buffer
                continue
                
            eligible.append(chunk)
        
        # Sort by (priority * age) to prioritize important, older memories
        eligible.sort(
            key=lambda c: c.priority * (current_time - c.timestamp),
            reverse=True
        )
        
        return eligible
    
    async def _consolidate_chunk(self, chunk: MemoryChunk) -> bool:
        """Consolidate a single chunk from working memory to long-term memory."""
        consolidated = False
        
        # Consolidate to episodic memory if available
        if self.episodic_memory is not None and self._is_episodic(chunk):
            await self._consolidate_to_episodic(chunk)
            consolidated = True
        
        # Extract semantic knowledge if semantic memory is available
        if self.semantic_memory is not None and self._contains_semantic_knowledge(chunk):
            await self._extract_semantic_knowledge(chunk)
            consolidated = True
        
        # If consolidated, we can remove from working memory
        if consolidated and chunk.expiration is None:
            # Set a short TTL to allow for any dependent processing
            chunk.expiration = time.time() + 10.0  # 10 seconds
        
        return consolidated
    
    def _is_episodic(self, chunk: MemoryChunk) -> bool:
        """Determine if a chunk should be stored in episodic memory."""
        # In a real implementation, this would use more sophisticated heuristics
        # For now, we'll use a simple content-based approach
        
        # Check content type
        if chunk.content_type not in ["text", "event"]:
            return False
            
        # Check for personal or experiential content
        personal_keywords = ["i ", "my ", "me ", "we ", "our ", "us "]
        content = str(chunk.content).lower()
        
        if any(kw in content for kw in personal_keywords):
            return True
            
        # Check metadata for episodic indicators
        episodic_indicators = ["memory", "experience", "event", "happened", "occurred"]
        for indicator in episodic_indicators:
            if indicator in content:
                return True
                
        return False
    
    async def _consolidate_to_episodic(self, chunk: MemoryChunk) -> str:
        """Consolidate a chunk to episodic memory."""
        if self.episodic_memory is None:
            raise ValueError("Episodic memory not available")
        
        # Prepare metadata
        metadata = chunk.metadata.copy()
        metadata.update({
            "source": chunk.source,
            "content_type": chunk.content_type,
            "original_priority": chunk.priority
        })
        
        # Add to episodic memory
        memory_id = await self.episodic_memory.add_memory(
            content=chunk.content,
            embedding=await self._generate_embedding(chunk.content),
            metadata=metadata,
            tags=list(chunk.metadata.get("tags", []) | {"consolidated", f"source:{chunk.source}"})
        )
        
        logger.debug(f"Consolidated to episodic memory: {memory_id}")
        return memory_id
    
    def _contains_semantic_knowledge(self, chunk: MemoryChunk) -> bool:
        """Determine if a chunk contains semantic knowledge that should be extracted."""
        # Simple heuristic: look for factual statements
        factual_indicators = [" is ", " are ", " has ", " have ", " can ", " cannot ",
                            " was ", " were ", " will ", " contains ", " includes "]
        
        content = f" {str(chunk.content).lower()} "
        return any(indicator in content for indicator in factual_indicators)
    
    async def _extract_semantic_knowledge(self, chunk: MemoryChunk) -> List[str]:
        """Extract semantic knowledge from a chunk and add to semantic memory."""
        if self.semantic_memory is None:
            raise ValueError("Semantic memory not available")
        
        # In a real implementation, this would use NLP to extract entities and relationships
        # For now, we'll use a simplified approach
        
        content = str(chunk.content)
        concept_ids = []
        
        # Simple pattern matching for concept extraction
        # This is a placeholder - in practice, you'd use NLP techniques
        if " is " in content.lower():
            parts = content.split(" is ", 1)
            if len(parts) == 2:
                concept_name = parts[0].strip()
                description = parts[1].strip(" .")
                
                # Add concept to semantic memory
                concept_id = await self.semantic_memory.add_concept(
                    name=concept_name,
                    description=description,
                    source=chunk.source,
                    confidence=chunk.priority,
                    tags=["extracted"]
                )
                concept_ids.append(concept_id)
                
                logger.debug(f"Extracted concept: {concept_name} -> {description}")
        
        return concept_ids
    
    async def _generate_embedding(self, content: Any) -> np.ndarray:
        """Generate an embedding for the given content."""
        # In a real implementation, this would use a pre-trained model
        # For now, return a random embedding
        return np.random.rand(384)

# Example usage
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    async def example():
        # Create memory systems
        working_memory = WorkingMemory()
        episodic_memory = EpisodicMemoryStore("episodic_memories.json")
        semantic_memory = SemanticMemory("semantic_memory.json")
        
        # Create consolidator
        config = ConsolidationConfig(
            strategy=ConsolidationStrategy.HYBRID,
            interval=60.0,  # 1 minute for testing
            min_chunk_age=5.0  # 5 seconds for testing
        )
        
        consolidator = MemoryConsolidator(
            working_memory=working_memory,
            episodic_memory=episodic_memory,
            semantic_memory=semantic_memory,
            config=config
        )
        
        # Add some test data to working memory
        await working_memory.add_chunk(
            content="I learned that Paris is the capital of France",
            priority=0.8,
            source="user_input",
            metadata={"type": "fact", "domain": "geography"}
        )
        
        await working_memory.add_chunk(
            content="Today I had pizza for lunch at Mario's Pizzeria",
            priority=0.9,
            source="user_input",
            metadata={"type": "experience", "meal": "lunch"}
        )
        
        # Start consolidation (would normally run in the background)
        await consolidator.consolidate(force=True)
        
        # Check what was consolidated
        print("Episodic memories:")
        for mem_id in episodic_memory.memories:
            mem = episodic_memory.memories[mem_id]
            print(f"- {mem.content} (tags: {mem.metadata.tags})")
        
        print("\nSemantic concepts:")
        for concept_id, concept in semantic_memory.concepts.items():
            print(f"- {concept.name}: {concept.description}")
    
    asyncio.run(example())
