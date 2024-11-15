from abc import ABC, abstractmethod
from typing import List, Callable, Optional, Any
from swarm import Agent

class BaseSwarmAgent(ABC):
    def __init__(
        self,
        name: str,
        instructions: str,
        functions: List[Callable],
    ):
        self._agent = Agent(
            name=name,
            instructions=instructions,
            functions=functions
        )
        
    @property
    def agent(self) -> Agent:
        return self._agent
    
    @property
    def name(self) -> str:
        return self._agent.name
    
    def update_functions(self, functions: List[Callable]) -> None:
        """Update the agent's available functions."""
        self._agent.functions = functions
        
    @abstractmethod
    def handle_query(self, query: str) -> Any:
        """Handle an incoming query."""
        pass
    
    def __call__(self, *args, **kwargs):
        return self._agent(*args, **kwargs)