import json
import os
from typing import Dict, Any
from pathlib import Path


class LocalLogger:
    """
    Simple class to read and write data to a JSON file.
    Uses an absolute file path at "ROOT_DIR/data/data.json".
    """

    def __init__(self):
        """Initialize the LocalLogger with a fixed file path."""
        # Get the absolute path to the project root
        root_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        # Create data directory if it doesn't exist
        data_dir = root_dir / "data"
        os.makedirs(data_dir, exist_ok=True)
        # Set the absolute file path
        self.file_path = data_dir / "data.json"
        print(f"LocalLogger initialized with file path: {self.file_path}")

    def read_json(self) -> Dict[str, Any]:
        """
        Read data from the JSON file.

        Returns:
            Dictionary containing the JSON data
        """
        try:
            if not os.path.exists(self.file_path):
                print(f"File not found: {self.file_path}, creating empty file")
                with open(self.file_path, "w") as f:
                    json.dump({}, f)
                return {}
            with open(self.file_path, "r") as f:
                data = json.load(f)
                return data
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {self.file_path}: {e}")
            return {}
        except Exception as e:
            print(f"Unexpected error reading {self.file_path}: {e}")
            return {}

    def write_json(self, data: Dict[str, Any]) -> None:
        """
        Write data to the JSON file, overwriting existing content.

        Args:
            data: Dictionary to serialize as JSON
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

        try:
            with open(self.file_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Data successfully written to {self.file_path}")
        except Exception as e:
            print(f"Error writing data to {self.file_path}: {e}")
