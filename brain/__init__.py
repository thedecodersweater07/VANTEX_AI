"""
VANTEX AI - Brain Module

This module serves as the central intelligence core of the VANTEX_AI system,
handling decision-making, task orchestration, and cognitive control.
"""

__version__ = "0.1.0"
__all__ = ['decision_engine', 'task_orchestrator', 'state_manager', 'cognitive_controller']

# Import core components
from .decision_engine import DecisionEngine
from .task_orchestrator import TaskOrchestrator
from .state_manager import StateManager
from .cognitive_controller import CognitiveController

class Brain:
    """Main brain class that initializes and coordinates all cognitive components."""
    
    def __init__(self, config=None):
        """Initialize the brain with configuration."""
        self.config = config or {}
        self.decision_engine = DecisionEngine()
        self.task_orchestrator = TaskOrchestrator()
        self.state_manager = StateManager()
        self.cognitive_controller = CognitiveController()
        
    def initialize(self):
        """Initialize all brain components."""
        self.decision_engine.initialize()
        self.task_orchestrator.initialize()
        self.state_manager.initialize()
        self.cognitive_controller.initialize()
        
    def process(self, input_data):
        """Process input data through the cognitive pipeline."""
        # Update state with new input
        self.state_manager.update_state(input_data)
        
        # Make decisions based on current state
        decisions = self.decision_engine.evaluate(self.state_manager.current_state)
        
        # Orchestrate tasks based on decisions
        tasks = self.task_orchestrator.create_tasks(decisions)
        
        # Execute tasks through cognitive controller
        results = self.cognitive_controller.execute_tasks(tasks)
        
        return results
