from interfaces.cli import SwarmCLI
from utils.logging import get_logger

logger = get_logger(__name__)

def main():
    try:
        cli = SwarmCLI()
        cli.run()
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
