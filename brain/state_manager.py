"""
State Manager Module

Manages the state and context of the VANTEX_AI system, providing a consistent
way to track, update, and observe system state across all components.
"""

import asyncio
import time
import json
from typing import Dict, Any, Optional, List, Set, Callable, Coroutine
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict
from uuid import uuid4

class StateType(Enum):
    """Types of states that can be managed."""
    SYSTEM = auto()      # System-level states (e.g., initialization, shutdown)
    TASK = auto()        # Task-related states
    MEMORY = auto()      # Memory and knowledge states
    SENSOR = auto()      # Sensor input states
    USER = auto()        # User interaction states
    LEARNING = auto()    # Learning and adaptation states

@dataclass
class StateUpdate:
    """Represents a state update event."""
    id: str = field(default_factory=lambda: str(uuid4()))
    namespace: str
    key: str
    old_value: Any
    new_value: Any
    timestamp: float = field(default_factory=time.time)
    source: str = "system"
    metadata: Dict[str, Any] = field(default_factory=dict)

class StateManager:
    """
    Centralized state management for the VANTEX_AI system.
    
    Features:
    - Namespaced state storage
    - State change observers
    - State history and versioning
    - Transaction support
    - State persistence
    """
    
    def __init__(self, persist_path: Optional[str] = None):
        """Initialize the state manager.
        
        Args:
            persist_path: Optional path to persist state (None for in-memory only)
        """
        self._state: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._history: Dict[str, List[StateUpdate]] = defaultdict(list)
        self._observers: Dict[str, List[Callable[[StateUpdate], Coroutine]]] = defaultdict(list)
        self._persist_path = persist_path
        self._lock = asyncio.Lock()
        self._initialized = False
        
        # Load persisted state if path is provided
        if self._persist_path:
            self._load_state()
    
    async def initialize(self):
        """Initialize the state manager."""
        if not self._initialized:
            # Initialize with default system state
            await self.set("system", "status", "initializing")
            await self.set("system", "startup_time", time.time())
            await self.set("system", "status", "running")
            self._initialized = True
    
    async def set(self, namespace: str, key: str, value: Any, 
                 source: str = "system", metadata: Optional[Dict] = None) -> StateUpdate:
        """Set a state value.
        
        Args:
            namespace: Namespace for the state (e.g., 'system', 'task', 'user')
            key: State key
            value: New value
            source: Source of the state change
            metadata: Additional metadata about the state change
            
        Returns:
            StateUpdate: The state update event
        """
        async with self._lock:
            old_value = self._state[namespace].get(key)
            
            # Skip if value hasn't changed
            if old_value == value:
                return StateUpdate(
                    namespace=namespace,
                    key=key,
                    old_value=old_value,
                    new_value=value,
                    source=source,
                    metadata=metadata or {}
                )
            
            # Update state
            self._state[namespace][key] = value
            
            # Create state update
            update = StateUpdate(
                namespace=namespace,
                key=key,
                old_value=old_value,
                new_value=value,
                source=source,
                metadata=metadata or {}
            )
            
            # Record in history
            self._history[f"{namespace}.{key}"].append(update)
            
            # Persist if configured
            if self._persist_path:
                self._save_state()
            
            # Notify observers
            await self._notify_observers(update)
            
            return update
    
    async def get(self, namespace: str, key: str, default: Any = None) -> Any:
        """Get a state value.
        
        Args:
            namespace: Namespace for the state
            key: State key
            default: Default value if key doesn't exist
            
        Returns:
            The state value or default if not found
        """
        return self._state.get(namespace, {}).get(key, default)
    
    async def get_namespace(self, namespace: str) -> Dict[str, Any]:
        """Get all key-value pairs in a namespace.
        
        Args:
            namespace: Namespace to retrieve
            
        Returns:
            Dictionary of key-value pairs in the namespace
        """
        return dict(self._state.get(namespace, {}))
    
    async def observe(self, pattern: str, callback: Callable[[StateUpdate], Coroutine]):
        """Register a callback to be called when a state matching the pattern changes.
        
        Args:
            pattern: Pattern to match state changes (e.g., 'system.*' or 'task.specific')
            callback: Async callback function that takes a StateUpdate
        """
        self._observers[pattern].append(callback)
    
    async def unobserve(self, pattern: str, callback: Callable[[StateUpdate], Coroutine]):
        """Unregister a previously registered observer."""
        if pattern in self._observers:
            if callback in self._observers[pattern]:
                self._observers[pattern].remove(callback)
    
    async def get_history(self, namespace: str, key: str, limit: int = 100) -> List[StateUpdate]:
        """Get history of state changes for a specific key.
        
        Args:
            namespace: Namespace of the state
            key: State key
            limit: Maximum number of history entries to return
            
        Returns:
            List of StateUpdate objects, most recent first
        """
        return self._history.get(f"{namespace}.{key}", [])[-limit:]
    
    async def transaction(self, updates: List[tuple], source: str = "system") -> List[StateUpdate]:
        """Execute multiple state updates as a single atomic transaction.
        
        Args:
            updates: List of (namespace, key, value, metadata) tuples
            source: Source of the state changes
            
        Returns:
            List of StateUpdate objects for each change
        """
        results = []
        
        async with self._lock:
            for update in updates:
                if len(update) == 3:
                    namespace, key, value = update
                    metadata = {}
                else:
                    namespace, key, value, metadata = update
                
                result = await self.set(namespace, key, value, source, metadata)
                results.append(result)
        
        return results
    
    async def _notify_observers(self, update: StateUpdate):
        """Notify all observers matching the updated state."""
        full_key = f"{update.namespace}.{update.key}"
        
        # Collect all matching patterns
        patterns_to_notify = []
        for pattern in self._observers:
            if self._pattern_matches(pattern, full_key):
                patterns_to_notify.append(pattern)
        
        # Notify all matching observers
        tasks = []
        for pattern in patterns_to_notify:
            for callback in self._observers[pattern]:
                tasks.append(callback(update))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def _pattern_matches(self, pattern: str, key: str) -> bool:
        """Check if a key matches a pattern with wildcards."""
        # Simple pattern matching with * as wildcard
        pattern_parts = pattern.split('.')
        key_parts = key.split('.')
        
        if len(pattern_parts) != len(key_parts):
            return False
        
        for p, k in zip(pattern_parts, key_parts):
            if p != '*' and p != k:
                return False
        
        return True
    
    def _save_state(self):
        """Save state to persistent storage."""
        if not self._persist_path:
            return
            
        try:
            with open(self._persist_path, 'w') as f:
                json.dump({
                    'state': self._state,
                    'history': {
                        k: [u.__dict__ for u in v] 
                        for k, v in self._history.items()
                    }
                }, f, default=str)
        except Exception as e:
            print(f"Error saving state: {e}")
    
    def _load_state(self):
        """Load state from persistent storage."""
        if not self._persist_path:
            return
            
        try:
            with open(self._persist_path, 'r') as f:
                data = json.load(f)
                self._state = defaultdict(dict, data.get('state', {}))
                
                # Convert history back to StateUpdate objects
                self._history = defaultdict(list)
                for k, updates in data.get('history', {}).items():
                    self._history[k] = [
                        StateUpdate(
                            id=u.get('id'),
                            namespace=u.get('namespace'),
                            key=u.get('key'),
                            old_value=u.get('old_value'),
                            new_value=u.get('new_value'),
                            timestamp=u.get('timestamp'),
                            source=u.get('source', 'system'),
                            metadata=u.get('metadata', {})
                        ) for u in updates
                    ]
        except FileNotFoundError:
            # No saved state, start fresh
            pass
        except Exception as e:
            print(f"Error loading state: {e}")

# Example usage
async def example_observer(update: StateUpdate):
    print(f"State changed: {update.namespace}.{update.key} = {update.new_value}")

async def main():
    # Create a state manager with persistence
    state_manager = StateManager("state.json")
    await state_manager.initialize()
    
    try:
        # Register an observer for all state changes
        await state_manager.observe("*.*", example_observer)
        
        # Set some states
        await state_manager.set("system", "status", "running")
        await state_manager.set("user", "current_task", "learning")
        
        # Update with metadata
        await state_manager.set("sensor", "temperature", 25.5, 
                              metadata={"unit": "celsius", "accuracy": 0.1})
        
        # Transaction example
        await state_manager.transaction([
            ("task", "progress", 0.1),
            ("task", "status", "in_progress", {"started_by": "system"})
        ])
        
        # Get state
        status = await state_manager.get("system", "status")
        print(f"System status: {status}")
        
        # Get history
        history = await state_manager.get_history("system", "status")
        print(f"Status history: {[h.new_value for h in history]}")
        
    finally:
        # State will be automatically saved to state.json
        pass

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
