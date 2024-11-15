import pytest
from unittest.mock import Mock

@pytest.fixture(autouse=True)
def mock_logger():
    """Mock logger for all tests."""
    with pytest.MonkeyPatch.context() as mp:
        mock = Mock()
        mp.setattr("utils.logging.get_logger", lambda _: mock)
        yield mock 