"""Script to start a new GEE session."""
import ee


def start() -> None:
    """Start a new session."""
    # Authenticate
    ee.Authenticate()

    # Initialize the library
    ee.Initialize()


if __name__ == "__main__":
    start()
