"""
Semantic Memory Module

Implements the semantic memory system for VANTEX_AI, responsible for storing and retrieving
factual knowledge, concepts, and their relationships in a structured knowledge graph.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
import uuid
import networkx as nx
import numpy as np

from ...utils.vector_store import VectorStore  # Will implement this later

class RelationshipType(Enum):
    """Types of relationships between concepts in the knowledge graph."""
    IS_A = "is_a"                    # Hierarchical relationship (e.g., "dog" IS_A "animal")
    HAS_PROPERTY = "has_property"    # Object-property relationship (e.g., "bird" HAS_PROPERTY "can fly")
    PART_OF = "part_of"              # Part-whole relationship (e.g., "wings" PART_OF "bird")
    USED_FOR = "used_for"            # Functional relationship (e.g., "hammer" USED_FOR "nailing")
    CAUSES = "causes"                # Causal relationship (e.g., "rain" CAUSES "wet ground")
    RELATED_TO = "related_to"        # General association
    INSTANCE_OF = "instance_of"      # Instance relationship (e.g., "Fido" INSTANCE_OF "dog")
    SYNONYM = "synonym"              # Similar or equivalent concepts
    OPPOSITE = "opposite"            # Opposite or antonym relationship
    PREREQUISITE = "prerequisite"    # Required knowledge or condition

@dataclass
class Concept:
    """Represents a concept in the semantic memory."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0           # Confidence in the concept's validity (0.0 to 1.0)
    last_updated: float = field(default_factory=time.time)
    source: str = "system"           # Source of this knowledge
    tags: Set[str] = field(default_factory=set)

@dataclass
class Relationship:
    """Represents a relationship between two concepts."""
    source_id: str
    target_id: str
    type: RelationshipType
    weight: float = 1.0               # Strength of the relationship (0.0 to 1.0)
    confidence: float = 1.0           # Confidence in the relationship (0.0 to 1.0)
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_updated: float = field(default_factory=time.time)

class SemanticMemory:
    """Manages a knowledge graph of concepts and their relationships."""
    
    def __init__(self, persist_path: Optional[str] = None):
        """Initialize the semantic memory.
        
        Args:
            persist_path: Path to persist the knowledge graph (None for in-memory only)
        """
        self.graph = nx.DiGraph()
        self.concepts: Dict[str, Concept] = {}
        self.vector_store = VectorStore(dimensions=384)  # For semantic search
        self.persist_path = persist_path
        self._lock = asyncio.Lock()
        
        if self.persist_path:
            self._load_knowledge_graph()
    
    async def add_concept(
        self,
        name: str,
        description: str = "",
        embedding: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        source: str = "system",
        confidence: float = 1.0
    ) -> str:
        """Add a new concept to the knowledge graph.
        
        Args:
            name: Name of the concept
            description: Description of the concept
            embedding: Optional vector representation
            metadata: Additional metadata
            tags: Optional list of tags
            source: Source of this knowledge
            confidence: Confidence in the concept's validity (0.0 to 1.0)
            
        Returns:
            str: ID of the created concept
        """
        # Check if concept with this name already exists
        existing_id = self._find_concept_by_name(name)
        if existing_id:
            # Update existing concept
            concept = self.concepts[existing_id]
            concept.description = description or concept.description
            concept.metadata.update(metadata or {})
            concept.tags.update(tags or [])
            concept.last_updated = time.time()
            concept.confidence = max(concept.confidence, confidence)
            
            # Update embedding if provided
            if embedding is not None:
                concept.embedding = embedding
                self.vector_store.update(existing_id, embedding)
                
            return existing_id
        
        # Create new concept
        concept = Concept(
            name=name,
            description=description,
            embedding=embedding,
            metadata=metadata or {},
            tags=set(tags or []),
            source=source,
            confidence=confidence
        )
        
        # Add to storage
        async with self._lock:
            self.concepts[concept.id] = concept
            self.graph.add_node(concept.id)
            
            # Add to vector store if embedding is available
            if concept.embedding is not None:
                self.vector_store.add(concept.id, concept.embedding)
            
            # Persist if configured
            if self.persist_path:
                self._save_knowledge_graph()
        
        return concept.id
    
    async def add_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: RelationshipType,
        weight: float = 1.0,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add a relationship between two concepts.
        
        Args:
            source_id: ID of the source concept
            target_id: ID of the target concept
            rel_type: Type of relationship
            weight: Strength of the relationship (0.0 to 1.0)
            confidence: Confidence in the relationship (0.0 to 1.0)
            metadata: Additional metadata
            
        Returns:
            bool: True if relationship was added, False otherwise
        """
        # Verify both concepts exist
        if source_id not in self.concepts or target_id not in self.concepts:
            return False
            
        # Create or update relationship
        rel_key = (source_id, target_id, rel_type.value)
        
        if self.graph.has_edge(source_id, target_id, key=rel_type.value):
            # Update existing relationship
            self.graph.edges[source_id, target_id, rel_type.value].update({
                'weight': weight,
                'confidence': confidence,
                'metadata': metadata or {},
                'last_updated': time.time()
            })
        else:
            # Add new relationship
            self.graph.add_edge(
                source_id,
                target_id,
                key=rel_type.value,
                type=rel_type,
                weight=weight,
                confidence=confidence,
                metadata=metadata or {},
                last_updated=time.time()
            )
        
        # Persist if configured
        if self.persist_path:
            self._save_knowledge_graph()
            
        return True
    
    async def get_concept(self, concept_id: str) -> Optional[Concept]:
        """Retrieve a concept by its ID."""
        return self.concepts.get(concept_id)
    
    async def find_concepts(
        self,
        query: Optional[str] = None,
        query_embedding: Optional[np.ndarray] = None,
        tags: Optional[List[str]] = None,
        min_confidence: float = 0.0,
        limit: int = 10
    ) -> List[Tuple[Concept, float]]:
        """Find concepts matching the query or embedding.
        
        Args:
            query: Text query (will be embedded if embedding is not provided)
            query_embedding: Optional vector for similarity search
            tags: Optional list of tags to filter by
            min_confidence: Minimum confidence threshold
            limit: Maximum number of results to return
            
        Returns:
            List of (concept, similarity_score) tuples, sorted by relevance
        """
        if query_embedding is None and query is not None:
            # In a real implementation, we would use an embedding model here
            # For now, we'll just do a simple text search
            query_embedding = np.random.rand(384)  # Placeholder
            
        if query_embedding is not None:
            # Vector similarity search
            similar_ids = self.vector_store.search(query_embedding, top_k=limit * 2)
            
            # Filter and sort results
            results = []
            for concept_id, similarity in similar_ids:
                concept = self.concepts.get(concept_id)
                if not concept or concept.confidence < min_confidence:
                    continue
                    
                # Apply tag filters
                if tags and not concept.tags.intersection(tags):
                    continue
                    
                results.append((concept, similarity))
                
                # Early termination if we have enough results
                if len(results) >= limit:
                    break
                    
            # Sort by similarity (descending)
            results.sort(key=lambda x: x[1], reverse=True)
            return results
            
        else:
            # Simple text search (fallback)
            results = []
            for concept in self.concepts.values():
                if (not query or query.lower() in concept.name.lower() or 
                    query.lower() in concept.description.lower()):
                    if concept.confidence >= min_confidence:
                        if not tags or concept.tags.intersection(tags):
                            # Simple relevance score based on name match
                            score = 0.0
                            if query:
                                if query.lower() in concept.name.lower():
                                    score += 0.7
                                if query.lower() in concept.description.lower():
                                    score += 0.3
                            results.append((concept, score))
            
            # Sort by relevance score
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:limit]
    
    async def get_related_concepts(
        self,
        concept_id: str,
        rel_type: Optional[RelationshipType] = None,
        direction: str = "out",
        min_confidence: float = 0.0
    ) -> List[Tuple[Concept, RelationshipType, float]]:
        """Get concepts related to the given concept.
        
        Args:
            concept_id: ID of the source concept
            rel_type: Optional relationship type to filter by
            direction: 'in', 'out', or 'both' for relationship direction
            min_confidence: Minimum confidence threshold for relationships
            
        Returns:
            List of (related_concept, relationship_type, weight) tuples
        """
        if concept_id not in self.concepts:
            return []
            
        results = []
        
        if direction in ("out", "both"):
            # Outgoing relationships
            for _, target_id, data in self.graph.out_edges(concept_id, data=True):
                if rel_type is None or data.get('type') == rel_type:
                    if data.get('confidence', 1.0) >= min_confidence:
                        results.append((
                            self.concepts[target_id],
                            data.get('type'),
                            data.get('weight', 1.0)
                        ))
                        
        if direction in ("in", "both"):
            # Incoming relationships
            for source_id, _, data in self.graph.in_edges(concept_id, data=True):
                if rel_type is None or data.get('type') == rel_type:
                    if data.get('confidence', 1.0) >= min_confidence:
                        results.append((
                            self.concepts[source_id],
                            data.get('type'),
                            data.get('weight', 1.0)
                        ))
        
        # Sort by relationship weight (descending)
        results.sort(key=lambda x: x[2], reverse=True)
        return results
    
    async def infer_relationships(
        self,
        concept_id: str,
        max_depth: int = 2,
        min_confidence: float = 0.0
    ) -> List[Tuple[Concept, RelationshipType, Concept, float]]:
        """Infer indirect relationships between concepts.
        
        Args:
            concept_id: ID of the source concept
            max_depth: Maximum depth to traverse the knowledge graph
            min_confidence: Minimum confidence threshold for inferred relationships
            
        Returns:
            List of (source_concept, relationship, target_concept, confidence) tuples
            representing inferred relationships
        """
        if concept_id not in self.concepts:
            return []
            
        # This is a simplified implementation
        # In a real system, you might use path finding, graph algorithms,
        # or machine learning to infer relationships
        
        # For now, we'll just return direct relationships
        results = []
        
        # Get direct relationships
        for target_id, data in self.graph[concept_id].items():
            for rel_data in data.values():
                if rel_data.get('confidence', 1.0) >= min_confidence:
                    results.append((
                        self.concepts[concept_id],
                        rel_data.get('type'),
                        self.concepts[target_id],
                        rel_data.get('confidence', 1.0)
                    ))
        
        return results
    
    def _find_concept_by_name(self, name: str) -> Optional[str]:
        """Find a concept by its name (case-insensitive)."""
        name_lower = name.lower()
        for concept_id, concept in self.concepts.items():
            if concept.name.lower() == name_lower:
                return concept_id
        return None
    
    def _save_knowledge_graph(self):
        """Save the knowledge graph to disk."""
        try:
            # Convert graph to a serializable format
            graph_data = {
                'concepts': {},
                'relationships': []
            }
            
            # Save concepts
            for concept_id, concept in self.concepts.items():
                concept_dict = {
                    'id': concept.id,
                    'name': concept.name,
                    'description': concept.description,
                    'embedding': concept.embedding.tolist() if concept.embedding is not None else None,
                    'metadata': concept.metadata,
                    'confidence': concept.confidence,
                    'last_updated': concept.last_updated,
                    'source': concept.source,
                    'tags': list(concept.tags)
                }
                graph_data['concepts'][concept_id] = concept_dict
            
            # Save relationships
            for source_id, target_id, data in self.graph.edges(data=True):
                for rel_type, rel_data in data.items():
                    if isinstance(rel_type, str):  # Skip the 'key' attribute
                        relationship = {
                            'source_id': source_id,
                            'target_id': target_id,
                            'type': rel_type,
                            'weight': rel_data.get('weight', 1.0),
                            'confidence': rel_data.get('confidence', 1.0),
                            'metadata': rel_data.get('metadata', {}),
                            'last_updated': rel_data.get('last_updated', time.time())
                        }
                        graph_data['relationships'].append(relationship)
            
            # Save to file
            with open(self.persist_path, 'w') as f:
                json.dump(graph_data, f, default=str)
                
        except Exception as e:
            print(f"Error saving knowledge graph: {e}")
    
    def _load_knowledge_graph(self):
        """Load the knowledge graph from disk."""
        try:
            with open(self.persist_path, 'r') as f:
                graph_data = json.load(f)
                
                # Load concepts
                self.concepts = {}
                for concept_id, concept_data in graph_data.get('concepts', {}).items():
                    concept = Concept(
                        id=concept_data['id'],
                        name=concept_data['name'],
                        description=concept_data['description'],
                        embedding=np.array(concept_data['embedding']) if concept_data['embedding'] else None,
                        metadata=concept_data.get('metadata', {}),
                        confidence=concept_data.get('confidence', 1.0),
                        last_updated=concept_data.get('last_updated', time.time()),
                        source=concept_data.get('source', 'system'),
                        tags=set(concept_data.get('tags', []))
                    )
                    self.concepts[concept_id] = concept
                    
                    # Add to vector store
                    if concept.embedding is not None:
                        self.vector_store.add(concept_id, concept.embedding)
                
                # Rebuild graph
                self.graph = nx.MultiDiGraph()
                
                # Add nodes
                for concept_id in self.concepts:
                    self.graph.add_node(concept_id)
                
                # Add edges
                for rel in graph_data.get('relationships', []):
                    self.graph.add_edge(
                        rel['source_id'],
                        rel['target_id'],
                        key=rel['type'],
                        type=rel['type'],
                        weight=rel.get('weight', 1.0),
                        confidence=rel.get('confidence', 1.0),
                        metadata=rel.get('metadata', {}),
                        last_updated=rel.get('last_updated', time.time())
                    )
                    
        except FileNotFoundError:
            # No saved knowledge graph yet
            pass
        except Exception as e:
            print(f"Error loading knowledge graph: {e}")

# Example usage
if __name__ == "__main__":
    import numpy as np
    
    # Create a semantic memory with persistence
    semantic_memory = SemanticMemory("semantic_memory.json")
    
    # Add some concepts
    dog_id = asyncio.run(semantic_memory.add_concept(
        name="dog",
        description="A domesticated carnivorous mammal",
        tags=["animal", "pet"],
        confidence=0.95
    ))
    
    animal_id = asyncio.run(semantic_memory.add_concept(
        name="animal",
        description="A living organism that feeds on organic matter",
        tags=["biology"],
        confidence=0.98
    ))
    
    # Add relationships
    asyncio.run(semantic_memory.add_relationship(
        source_id=dog_id,
        target_id=animal_id,
        rel_type=RelationshipType.IS_A,
        weight=0.99,
        confidence=0.98
    ))
    
    # Find concepts
    results = asyncio.run(semantic_memory.find_concepts(query="dog"))
    for concept, score in results:
        print(f"Found concept: {concept.name} (score: {score:.2f})")
    
    # Get related concepts
    related = asyncio.run(semantic_memory.get_related_concepts(
        concept_id=dog_id,
        rel_type=RelationshipType.IS_A
    ))
    
    for concept, rel_type, weight in related:
        print(f"{semantic_memory.concepts[dog_id].name} {rel_type.value} {concept.name} (weight: {weight:.2f})")
