"""
Task Orchestrator Module

Manages the creation, scheduling, and execution of tasks within the VANTEX_AI system.
Handles task dependencies, prioritization, and resource allocation.
"""

import asyncio
import time
from typing import Dict, List, Optional, Set, Any, Callable, Coroutine
from dataclasses import dataclass, field
from enum import Enum, auto
from uuid import uuid4

class TaskStatus(Enum):
    """Possible states of a task."""
    PENDING = auto()
    READY = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()

class TaskPriority(Enum):
    """Priority levels for task scheduling."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4

@dataclass(order=True)
class Task:
    """Represents a unit of work in the VANTEX_AI system."""
    # Basic task information
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    created_at: float = field(default_factory=time.time)
    
    # Task configuration
    priority: TaskPriority = TaskPriority.NORMAL
    timeout: Optional[float] = None  # in seconds
    max_retries: int = 0
    retry_count: int = 0
    
    # Task dependencies and relationships
    dependencies: Set[str] = field(default_factory=set)  # Task IDs this task depends on
    dependents: Set[str] = field(default_factory=set)  # Tasks that depend on this one
    
    # Task execution
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[Exception] = None
    
    # Execution context
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Callable to execute
    coro: Optional[Callable[..., Coroutine]] = None
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.name:
            self.name = f"Task-{self.id[:8]}"
    
    def add_dependency(self, task_id: str):
        """Add a task ID that this task depends on."""
        self.dependencies.add(task_id)
    
    def remove_dependency(self, task_id: str) -> bool:
        """Remove a task dependency. Returns True if the task was a dependency."""
        if task_id in self.dependencies:
            self.dependencies.remove(task_id)
            return True
        return False
    
    def is_ready(self) -> bool:
        """Check if all dependencies are satisfied."""
        return not self.dependencies and self.status == TaskStatus.PENDING
    
    def mark_completed(self, result: Any = None):
        """Mark the task as completed with an optional result."""
        self.status = TaskStatus.COMPLETED
        self.result = result
        self.completed_at = time.time()
    
    def mark_failed(self, error: Exception):
        """Mark the task as failed with an error."""
        self.status = TaskStatus.FAILED
        self.error = error
        self.completed_at = time.time()

class TaskOrchestrator:
    """
    Manages the execution of tasks with support for dependencies, prioritization,
    and resource management.
    """
    
    def __init__(self, max_workers: int = 10):
        """Initialize the task orchestrator."""
        self.max_workers = max_workers
        self.tasks: Dict[str, Task] = {}
        self.task_queue = asyncio.PriorityQueue()
        self.workers: Set[asyncio.Task] = set()
        self.task_completion_events: Dict[str, asyncio.Event] = {}
        self.running = False
        self._lock = asyncio.Lock()
        
    async def initialize(self):
        """Initialize the task orchestrator and start worker tasks."""
        if not self.running:
            self.running = True
            for _ in range(self.max_workers):
                worker = asyncio.create_task(self._worker_loop())
                self.workers.add(worker)
                worker.add_done_callback(self.workers.remove)
    
    async def shutdown(self):
        """Gracefully shut down the task orchestrator."""
        self.running = False
        
        # Cancel all worker tasks
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to complete
        if self.workers:
            await asyncio.wait(self.workers)
    
    async def create_task(
        self,
        coro: Callable[..., Coroutine],
        name: str = "",
        description: str = "",
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: Optional[float] = None,
        max_retries: int = 0,
        dependencies: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """
        Create and schedule a new task.
        
        Args:
            coro: The coroutine to execute
            name: Human-readable task name
            description: Task description
            priority: Task priority
            timeout: Maximum execution time in seconds
            max_retries: Maximum number of retry attempts
            dependencies: List of task IDs this task depends on
            **kwargs: Additional arguments to pass to the coroutine
            
        Returns:
            str: The ID of the created task
        """
        task = Task(
            name=name,
            description=description,
            priority=priority,
            timeout=timeout,
            max_retries=max_retries,
            coro=coro,
            kwargs=kwargs,
        )
        
        # Set up dependencies
        if dependencies:
            task.dependencies.update(dependencies)
            # We'll set up the reverse dependencies when adding the task
        
        async with self._lock:
            self.tasks[task.id] = task
            self.task_completion_events[task.id] = asyncio.Event()
            
            # Set up reverse dependencies
            for dep_id in task.dependencies:
                if dep_id in self.tasks:
                    self.tasks[dep_id].dependents.add(task.id)
            
            # If no dependencies, add to queue
            if not task.dependencies:
                await self._enqueue_task(task.id)
        
        return task.id
    
    async def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Wait for a task to complete and return its result."""
        if task_id not in self.task_completion_events:
            raise ValueError(f"Unknown task ID: {task_id}")
        
        await asyncio.wait_for(self.task_completion_events[task_id].wait(), timeout=timeout)
        
        task = self.tasks[task_id]
        if task.status == TaskStatus.FAILED and task.error:
            raise task.error
        
        return task.result
    
    async def _enqueue_task(self, task_id: str):
        """Add a task to the execution queue."""
        task = self.tasks[task_id]
        if task.status != TaskStatus.PENDING:
            return
            
        task.status = TaskStatus.READY
        priority = (task.priority.value, task.created_at)
        await self.task_queue.put((priority, task_id))
    
    async def _worker_loop(self):
        """Worker coroutine that processes tasks from the queue."""
        while self.running:
            try:
                # Get the next task from the queue
                try:
                    _, task_id = await asyncio.wait_for(
                        self.task_queue.get(),
                        timeout=0.1
                    )
                except asyncio.TimeoutError:
                    continue
                
                task = self.tasks.get(task_id)
                if not task or task.status != TaskStatus.READY:
                    self.task_queue.task_done()
                    continue
                
                # Mark task as running
                task.status = TaskStatus.RUNNING
                
                # Execute the task
                try:
                    if task.coro:
                        result = await asyncio.wait_for(
                            task.coro(*task.args, **task.kwargs),
                            timeout=task.timeout
                        )
                        task.mark_completed(result)
                except asyncio.CancelledError:
                    task.status = TaskStatus.CANCELLED
                    raise
                except Exception as e:
                    if task.retry_count < task.max_retries:
                        # Retry the task
                        task.retry_count += 1
                        task.status = TaskStatus.PENDING
                        await self._enqueue_task(task_id)
                    else:
                        task.mark_failed(e)
                finally:
                    self.task_queue.task_done()
                    
                    # Notify any waiting tasks
                    self.task_completion_events[task_id].set()
                    
                    # Check for dependent tasks that might now be ready
                    await self._check_dependent_tasks(task_id)
                    
            except Exception as e:
                # Log the error but keep the worker alive
                print(f"Error in worker: {e}")
    
    async def _check_dependent_tasks(self, task_id: str):
        """Check if any tasks are waiting on the completed task."""
        async with self._lock:
            task = self.tasks.get(task_id)
            if not task:
                return
                
            for dep_id in task.dependents:
                if dep_id not in self.tasks:
                    continue
                    
                dependent = self.tasks[dep_id]
                dependent.dependencies.discard(task_id)
                
                if not dependent.dependencies and dependent.status == TaskStatus.PENDING:
                    await self._enqueue_task(dep_id)

# Example usage
async def example_task(name: str, duration: int = 1):
    print(f"Starting task {name}")
    await asyncio.sleep(duration)
    print(f"Completed task {name}")
    return f"Result of {name}"

async def main():
    # Create an orchestrator with 3 workers
    orchestrator = TaskOrchestrator(max_workers=3)
    await orchestrator.initialize()
    
    try:
        # Create some tasks with dependencies
        task1_id = await orchestrator.create_task(
            example_task, "Task 1", "First task", TaskPriority.HIGH, 5, 0
        )
        
        task2_id = await orchestrator.create_task(
            example_task, "Task 2", "Depends on Task 1", TaskPriority.NORMAL, 2, 0,
            dependencies=[task1_id]
        )
        
        task3_id = await orchestrator.create_task(
            example_task, "Task 3", "Depends on Task 1", TaskPriority.NORMAL, 1, 0,
            dependencies=[task1_id]
        )
        
        task4_id = await orchestrator.create_task(
            example_task, "Task 4", "Depends on Task 2 and 3", TaskPriority.NORMAL, 1, 0,
            dependencies=[task2_id, task3_id]
        )
        
        # Wait for all tasks to complete
        await orchestrator.wait_for_task(task4_id)
        print("All tasks completed!")
        
    finally:
        await orchestrator.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
