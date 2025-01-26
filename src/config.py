import os

# Testing configuration
TESTING = os.getenv("TESTING", "false").lower() == "true"

# Model configuration
if TESTING:
    MODEL_DIR = "tests/fixtures/models"
else:
    MODEL_DIR = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "models"
    )