import os
import shutil
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def setup_test_env():
    """Set up test environment."""
    # Set testing environment
    os.environ["TESTING"] = "true"
    
    # Create test directories
    test_dir = Path("tests/fixtures")
    model_dir = test_dir / "models" / "latest"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    yield
    
    # Cleanup
    shutil.rmtree(test_dir, ignore_errors=True)
    os.environ.pop("TESTING", None)