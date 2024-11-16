from typing import List
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from swarm import Swarm

from utils.logging import get_logger
from core.agents.base import BaseSwarmAgent
from core.agents.coordinator import CoordinatorAgent
from core.agents.sql_agent import SQLAgent
from core.agents.rag_agent import RAGAgent

logger = get_logger(__name__)

class SwarmCLI:
    def __init__(self):
        self.console = Console()
        self.client = Swarm()
        self.messages: List[dict] = []
        
        self.sql_agent = SQLAgent()
        self.rag_agent = RAGAgent()
        self.coordinator = CoordinatorAgent()
        self.coordinator.set_transfer_functions(self.sql_agent, self.rag_agent)
        
    def print_message(self, message: dict) -> None:
        sender = message.get("sender", message.get("role", "unknown"))
        content = message.get("content", "")
        
        if not content:
            return
            
        styles = {
            "user": ("bold blue", "📝 User"),
            "Coordinator": ("bold yellow", "🎯 Coordinator"),
            "SQL Agent": ("bold green", "💾 SQL Agent"),
            "RAG Agent": ("bold magenta", "📚 RAG Agent"),
            "unknown": ("bold white", sender)
        }
        
        style, title = styles.get(sender, styles["unknown"])
        
        panel = Panel(
            Markdown(content) if sender != "user" else content,
            title=title,
            style=style,
            border_style=style,
            padding=(1, 2)
        )
        self.console.print(panel)
        
    def print_conversation(self) -> None:
        current_turn = []
        
        for message in self.messages:
            if message.get("content") is None:
                continue
            
            sender = message.get("sender", message.get("role", "unknown"))
            content = message.get("content", "")
            
            if sender == "user":
                if current_turn:
                    self.console.print("─" * 80)
                    for turn_message in current_turn:
                        self.print_message(turn_message)
                    current_turn = []
                current_turn.append({"sender": sender, "content": content})
            else:
                current_turn.append({"sender": sender, "content": content})
        
        if current_turn:
            self.console.print("─" * 80)
            for turn_message in current_turn:
                self.print_message(turn_message)
                
    def run(self) -> None:
        try:
            self.console.print("[bold magenta]Welcome to the Swarm CLI![/bold magenta]")
            self.console.print("[bold]Type 'quit' to exit[/bold]")
            
            agent = self.coordinator
            
            while True:
                user_input = self.console.input("\n[bold blue]Enter your question:[/bold blue] ").strip()
                
                if user_input.lower() == "quit":
                    self.console.print("\n[bold yellow]Gracefully shutting down...[/bold yellow]")
                    break
                    
                if user_input.lower() == "clear":
                    self.console.clear()
                    self.messages = []
                    continue
                
                self.messages.append({"role": "user", "content": user_input})
                
                try:
                    with self.console.status(
                        "[bold cyan]🤔 Processing your query...[/bold cyan]", 
                        spinner="dots12"
                    ):
                        response = self.client.run(agent=agent.agent, messages=self.messages)
                        self.messages = response.messages
                        agent = self._get_agent_from_response(response.agent)
                        
                    self.print_conversation()
                    
                except Exception as e:
                    logger.error(f"Error processing query: {str(e)}")
                    self.console.print(f"[bold red]Error:[/bold red] {str(e)}")
                    
        except KeyboardInterrupt:
            self.console.print("\n[bold yellow]Gracefully shutting down...[/bold yellow]")
        except Exception as e:
            logger.error(f"Unexpected error in CLI: {str(e)}")
            self.console.print(f"[bold red]Unexpected error:[/bold red] {str(e)}")
            
    def _get_agent_from_response(self, agent) -> BaseSwarmAgent:
        if agent is None:
            return self.coordinator
        
        if isinstance(agent, type) and issubclass(agent, BaseSwarmAgent):
            if agent.__name__ == "SQLAgent":
                return self.sql_agent
            elif agent.__name__ == "RAGAgent":
                return self.rag_agent
            return self.coordinator
        
        if isinstance(agent, BaseSwarmAgent):
            return agent
            
        agent_name = getattr(agent, 'name', None)
        if agent_name:
            name_map = {
                "Coordinator": self.coordinator,
                "SQL Agent": self.sql_agent,
                "RAG Agent": self.rag_agent
            }
            return name_map.get(agent_name, self.coordinator)
            
        return self.coordinator
