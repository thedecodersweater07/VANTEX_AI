"""
Cognitive Controller Module

Coordinates and manages the cognitive functions of the VANTEX_AI system,
acting as the central control unit that integrates various cognitive components.
"""

import asyncio
import time
from typing import Dict, Any, Optional, List, Set, Callable, Coroutine, Type, TypeVar
from dataclasses import dataclass, field
from enum import Enum, auto
import logging

# Import other brain components
from .state_manager import StateManager, StateUpdate
from .task_orchestrator import Task, TaskOrchestrator, TaskStatus, TaskPriority
from .decision_engine import DecisionEngine, DecisionNode, DecisionType

# Type variable for component types
T = TypeVar('T')

class CognitiveState(Enum):
    """Possible states of the cognitive controller."""
    BOOTING = auto()
    INITIALIZING = auto()
    READY = auto()
    PROCESSING = auto()
    LEARNING = auto()
    SHUTTING_DOWN = auto()
    ERROR = auto()

@dataclass
class CognitiveModule:
    """Base class for all cognitive modules."""
    name: str
    dependencies: Set[str] = field(default_factory=set)
    priority: int = 0
    is_essential: bool = True
    
    async def initialize(self):
        """Initialize the module."""
        pass
        
    async def process(self, input_data: Any) -> Any:
        """Process input data and return result."""
        raise NotImplementedError
        
    async def shutdown(self):
        """Clean up resources."""
        pass

class CognitiveController:
    """
    Central controller for cognitive functions in the VANTEX_AI system.
    
    Manages the lifecycle of cognitive modules, coordinates their interactions,
    and ensures proper data flow between components.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the cognitive controller."""
        self.config = config or {}
        self.logger = self._setup_logging()
        
        # Core components
        self.state_manager = StateManager()
        self.task_orchestrator = TaskOrchestrator()
        self.decision_engine = DecisionEngine()
        
        # Module registry
        self.modules: Dict[str, CognitiveModule] = {}
        self.module_initialized: Set[str] = set()
        self.module_dependencies: Dict[str, Set[str]] = {}
        
        # State
        self._state = CognitiveState.BOOTING
        self._initialized = False
        self._shutdown_event = asyncio.Event()
        self._module_lock = asyncio.Lock()
        
        # Register state update handler
        self.state_manager.observe("system.*", self._handle_system_state_change)
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the cognitive controller."""
        logger = logging.getLogger("cognitive_controller")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    async def initialize(self):
        """Initialize the cognitive controller and all registered modules."""
        if self._initialized:
            return
            
        self.logger.info("Initializing Cognitive Controller...")
        self._state = CognitiveState.INITIALIZING
        
        try:
            # Initialize core components
            await self.state_manager.initialize()
            await self.task_orchestrator.initialize()
            await self.decision_engine.initialize()
            
            # Initialize modules in dependency order
            await self._initialize_modules()
            
            # Set initial state
            self._state = CognitiveState.READY
            self._initialized = True
            self.logger.info("Cognitive Controller initialized successfully")
            
            # Start main processing loop
            asyncio.create_task(self._main_loop())
            
        except Exception as e:
            self._state = CognitiveState.ERROR
            self.logger.error(f"Failed to initialize Cognitive Controller: {e}", exc_info=True)
            raise
    
    async def register_module(self, module: CognitiveModule):
        """Register a cognitive module with the controller."""
        if module.name in self.modules:
            raise ValueError(f"Module '{module.name}' is already registered")
            
        async with self._module_lock:
            self.modules[module.name] = module
            self.module_dependencies[module.name] = set(module.dependencies)
            
            # Check for circular dependencies
            if self._has_circular_dependency(module.name):
                del self.modules[module.name]
                del self.module_dependencies[module.name]
                raise ValueError(f"Circular dependency detected for module '{module.name}'")
            
            self.logger.info(f"Registered module: {module.name}")
    
    async def get_module(self, name: str) -> Optional[CognitiveModule]:
        """Get a registered module by name."""
        return self.modules.get(name)
    
    async def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data through the cognitive pipeline."""
        if not self._initialized:
            raise RuntimeError("Cognitive Controller not initialized")
            
        self._state = CognitiveState.PROCESSING
        results = {}
        
        try:
            # Update state with input data
            await self.state_manager.set("input", "last_input", input_data)
            
            # Create a decision node based on input
            decision_node = await self._create_decision_node(input_data)
            
            # Get decisions from the decision engine
            decisions = await self.decision_engine.evaluate(decision_node)
            
            # Create tasks based on decisions
            tasks = []
            for decision_id, decision in decisions.items():
                if decision["confidence"] > 0.5:  # Only proceed with high-confidence decisions
                    task = await self._create_processing_task(decision_id, decision, input_data)
                    if task:
                        tasks.append(task)
            
            # Execute tasks in parallel
            if tasks:
                task_results = await asyncio.gather(
                    *(self._execute_task(task) for task in tasks),
                    return_exceptions=True
                )
                
                # Process results
                for i, result in enumerate(task_results):
                    if isinstance(result, Exception):
                        self.logger.error(
                            f"Error in task {tasks[i].name}: {result}",
                            exc_info=result
                        )
                    else:
                        results[tasks[i].name] = result
            
            # Update state with results
            await self.state_manager.set("output", "last_results", results)
            
            return results
            
        except Exception as e:
            self._state = CognitiveState.ERROR
            self.logger.error(f"Error processing input: {e}", exc_info=True)
            raise
            
        finally:
            self._state = CognitiveState.READY
    
    async def shutdown(self):
        """Shut down the cognitive controller and all modules."""
        if not self._initialized:
            return
            
        self.logger.info("Shutting down Cognitive Controller...")
        self._state = CognitiveState.SHUTTING_DOWN
        self._shutdown_event.set()
        
        # Shutdown modules in reverse initialization order
        for module_name in reversed(list(self.module_initialized)):
            try:
                await self.modules[module_name].shutdown()
                self.logger.info(f"Shut down module: {module_name}")
            except Exception as e:
                self.logger.error(f"Error shutting down module {module_name}: {e}", exc_info=True)
        
        # Shutdown core components
        try:
            await self.task_orchestrator.shutdown()
            await self.state_manager.set("system", "status", "shutdown")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}", exc_info=True)
        
        self._initialized = False
        self.logger.info("Cognitive Controller shutdown complete")
    
    async def _initialize_modules(self):
        """Initialize all registered modules in dependency order."""
        initialized = set()
        remaining = set(self.modules.keys())
        
        while remaining:
            progress = False
            
            # Try to initialize modules with no uninitialized dependencies
            for module_name in list(remaining):
                module = self.modules[module_name]
                
                # Check if all dependencies are initialized
                deps_initialized = all(
                    dep in initialized or dep not in self.modules
                    for dep in module.dependencies
                )
                
                if deps_initialized:
                    try:
                        self.logger.info(f"Initializing module: {module_name}")
                        await module.initialize()
                        initialized.add(module_name)
                        remaining.remove(module_name)
                        progress = True
                        self.logger.info(f"Initialized module: {module_name}")
                    except Exception as e:
                        if module.is_essential:
                            self.logger.error(
                                f"Failed to initialize essential module {module_name}: {e}",
                                exc_info=True
                            )
                            # For essential modules, re-raise the exception
                            raise
                        else:
                            self.logger.warning(
                                f"Failed to initialize non-essential module {module_name}: {e}",
                                exc_info=True
                            )
                            # For non-essential modules, skip and continue
                            initialized.add(module_name)
                            remaining.remove(module_name)
                            progress = True
            
            # If no progress was made, we have a circular dependency
            if not progress and remaining:
                raise RuntimeError(
                    f"Circular dependency or missing dependencies detected. "
                    f"Remaining modules: {remaining}"
                )
        
        self.module_initialized = initialized
    
    async def _create_decision_node(self, input_data: Dict[str, Any]) -> DecisionNode:
        """Create a decision node based on input data."""
        # This is a simplified example - in practice, this would be more sophisticated
        return DecisionNode(
            id=f"input_{int(time.time())}",
            decision_type=DecisionType.ACTION_SELECTION,
            criteria={
                "relevance": 0.4,
                "complexity": 0.3,
                "priority": 0.3
            },
            options=[
                {"action": "process_text", "relevance": 0.8, "complexity": 0.6, "priority": 0.7},
                {"action": "process_image", "relevance": 0.5, "complexity": 0.8, "priority": 0.5},
                {"action": "delegate", "relevance": 0.9, "complexity": 0.3, "priority": 0.9}
            ],
            constraints={"max_complexity": 0.8},
            context={"input_type": type(input_data).__name__}
        )
    
    async def _create_processing_task(
        self, 
        decision_id: str, 
        decision: Dict[str, Any], 
        input_data: Any
    ) -> Optional[Task]:
        """Create a task based on a decision."""
        action = decision["decision"].get("action")
        
        if not action:
            return None
            
        # Map action to module and method
        module_name, method_name = self._map_action_to_module(action)
        if not module_name or not method_name:
            self.logger.warning(f"No handler for action: {action}")
            return None
            
        # Create task
        return Task(
            name=f"{module_name}.{method_name}",
            coro=self._execute_module_method,
            kwargs={
                "module_name": module_name,
                "method_name": method_name,
                "input_data": input_data,
                "decision": decision
            },
            priority=TaskPriority.NORMAL,
            timeout=30.0  # Default timeout of 30 seconds
        )
    
    def _map_action_to_module(self, action: str) -> tuple[Optional[str], Optional[str]]:
        """Map an action to a module and method."""
        # This is a simplified mapping - in practice, this would be more sophisticated
        action_map = {
            "process_text": ("nlp_processor", "process_text"),
            "process_image": ("vision_processor", "process_image"),
            "delegate": ("delegation_manager", "delegate_task")
        }
        return action_map.get(action, (None, None))
    
    async def _execute_task(self, task: Task) -> Any:
        """Execute a task and handle errors."""
        try:
            self.logger.info(f"Executing task: {task.name}")
            result = await task.coro(*task.args, **task.kwargs)
            self.logger.info(f"Completed task: {task.name}")
            return result
        except Exception as e:
            self.logger.error(f"Error in task {task.name}: {e}", exc_info=True)
            raise
    
    async def _execute_module_method(
        self, 
        module_name: str, 
        method_name: str, 
        input_data: Any,
        decision: Dict[str, Any]
    ) -> Any:
        """Execute a method on a module with proper error handling."""
        module = self.modules.get(module_name)
        if not module:
            raise ValueError(f"Module not found: {module_name}")
            
        method = getattr(module, method_name, None)
        if not method or not callable(method):
            raise ValueError(f"Method not found or not callable: {module_name}.{method_name}")
            
        # Call the method with input data and decision context
        return await method(input_data, decision_context=decision)
    
    async def _main_loop(self):
        """Main processing loop for the cognitive controller."""
        self.logger.info("Starting main processing loop")
        
        while not self._shutdown_event.is_set() and self._initialized:
            try:
                # Check for new input in the state manager
                last_input = await self.state_manager.get("input", "last_input")
                last_processed = await self.state_manager.get("system", "last_processed_input")
                
                if last_input and last_input != last_processed:
                    # Process the new input
                    await self.process_input(last_input)
                    await self.state_manager.set("system", "last_processed_input", last_input)
                
                # Small sleep to prevent busy waiting
                await asyncio.sleep(0.1)
                
            except asyncio.CancelledError:
                self.logger.info("Main loop cancelled")
                break
                
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}", exc_info=True)
                await asyncio.sleep(1)  # Prevent tight error loop
    
    def _has_circular_dependency(self, module_name: str, path: Optional[List[str]] = None) -> bool:
        """Check if adding a module would create a circular dependency."""
        if path is None:
            path = []
            
        if module_name in path:
            return True
            
        path = path + [module_name]
        
        for dep in self.module_dependencies.get(module_name, []):
            if dep in self.modules and self._has_circular_dependency(dep, path):
                return True
                
        return False
    
    async def _handle_system_state_change(self, update: StateUpdate):
        """Handle changes to system state."""
        if update.namespace == "system" and update.key == "shutdown_requested":
            if update.new_value:
                self.logger.info("Shutdown requested, initiating shutdown sequence...")
                await self.shutdown()

# Example usage
if __name__ == "__main__":
    import asyncio
    
    # Example module
    class ExampleModule(CognitiveModule):
        def __init__(self):
            super().__init__("example", dependencies=[])
            
        async def process_text(self, text: str, decision_context: Optional[Dict] = None) -> str:
            print(f"Processing text: {text}")
            return f"Processed: {text.upper()}"
    
    async def main():
        # Create and initialize the cognitive controller
        controller = CognitiveController()
        
        try:
            # Register modules
            await controller.register_module(ExampleModule())
            
            # Initialize the controller
            await controller.initialize()
            
            # Process some input
            result = await controller.process_input({"type": "text", "content": "Hello, VANTEX_AI!"})
            print(f"Processing result: {result}")
            
            # Keep running until interrupted
            while True:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            await controller.shutdown()
    
    asyncio.run(main())
