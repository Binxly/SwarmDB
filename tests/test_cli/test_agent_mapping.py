import pytest
from unittest.mock import Mock, patch

from interfaces.cli import SwarmCLI

@pytest.fixture
def cli():
    with patch('interfaces.cli.Console'), \
         patch('interfaces.cli.Swarm'):
        return SwarmCLI()

class TestAgentMapping:
    def test_get_agent_from_response_coordinator(self, cli):
        """Test mapping coordinator agent."""
        response_agent = cli.coordinator.agent
        mapped_agent = cli._get_agent_from_response(response_agent)
        assert mapped_agent == cli.coordinator
        
    def test_get_agent_from_response_sql(self, cli):
        """Test mapping SQL agent."""
        response_agent = cli.sql_agent.agent
        mapped_agent = cli._get_agent_from_response(response_agent)
        assert mapped_agent == cli.sql_agent
        
    def test_get_agent_from_response_rag(self, cli):
        """Test mapping RAG agent."""
        response_agent = cli.rag_agent.agent
        mapped_agent = cli._get_agent_from_response(response_agent)
        assert mapped_agent == cli.rag_agent
        
    def test_get_agent_from_response_unknown(self, cli):
        """Test mapping unknown agent defaults to coordinator."""
        unknown_agent = Mock()
        mapped_agent = cli._get_agent_from_response(unknown_agent)
        assert mapped_agent == cli.coordinator 