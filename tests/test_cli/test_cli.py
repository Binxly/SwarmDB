from unittest.mock import Mock, patch
import pytest
from rich.console import Console
from swarm import Swarm, Agent

from interfaces.cli import SwarmCLI
from core.agents.coordinator import CoordinatorAgent
from core.agents.sql_agent import SQLAgent
from core.agents.rag_agent import RAGAgent

@pytest.fixture
def mock_console():
    console = Mock(spec=Console)
    console.print = Mock()
    console.input = Mock()
    return console

@pytest.fixture
def mock_swarm():
    return Mock(spec=Swarm)

@pytest.fixture
def cli(mock_console, mock_swarm):
    with patch('interfaces.cli.Console', return_value=mock_console), \
         patch('interfaces.cli.Swarm', return_value=mock_swarm):
        cli = SwarmCLI()
        return cli

class TestSwarmCLI:
    def test_initialization(self, cli):
        """Test that CLI initializes with all required components."""
        assert cli.coordinator is not None
        assert cli.sql_agent is not None
        assert cli.rag_agent is not None
        assert isinstance(cli.messages, list)
        
    def test_print_message_user(self, cli, mock_console):
        """Test printing user message."""
        message = {"role": "user", "content": "test message"}
        cli.print_message(message)
        mock_console.print.assert_called_once()
        
    def test_print_message_empty_content(self, cli, mock_console):
        """Test handling empty message content."""
        message = {"role": "user", "content": ""}
        cli.print_message(message)
        mock_console.print.assert_not_called()
        
    def test_print_conversation(self, cli, mock_console):
        """Test conversation printing."""
        messages = [
            {"role": "user", "content": "test question"},
            {"sender": "Coordinator", "content": "test response"}
        ]
        cli.messages = messages
        cli.print_conversation()
        assert mock_console.print.call_count >= 2
        
    @pytest.mark.parametrize("command,expected_calls", [
        ("quit", 0),
        ("clear", 1),
        ("test question", 1)
    ])
    def test_run_commands(self, cli, mock_console, mock_swarm, command, expected_calls):
        """Test different CLI commands."""
        mock_console.input.return_value = command
        mock_swarm.run.return_value = Mock(
            messages=[{"role": "assistant", "content": "test"}],
            agent=cli.coordinator.agent
        )
        
        if command == "quit":
            cli.run()
            assert mock_swarm.run.call_count == 0
        else:
            with patch.object(cli, 'print_conversation') as mock_print:
                cli.run()
                assert mock_print.call_count == expected_calls
                
    def test_error_handling(self, cli, mock_console, mock_swarm):
        """Test error handling in query processing."""
        mock_console.input.return_value = "test question"
        mock_swarm.run.side_effect = Exception("Test error")
        
        cli.run()
        
        error_calls = [
            call for call in mock_console.print.call_args_list 
            if "Error" in str(call)
        ]
        assert len(error_calls) > 0 