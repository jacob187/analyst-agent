import json
from typing import Dict, Any


class LocalLogger:
    """
    Simple class to read and write data to a JSON file.
    Uses a fixed file path at "../data/data.json".
    """

    def __init__(self):
        """Initialize the LocalLogger with a fixed file path."""
        self.file_path = "../data/data.json"

    def read_json(self) -> Dict[str, Any]:
        """
        Read data from the JSON file.

        Returns:
            Dictionary containing the JSON data
        """
        with open(self.file_path, "r") as f:
            return json.load(f)

    def write_json(self, data: Dict[str, Any]) -> None:
        """
        Write data to the JSON file, overwriting existing content.

        Args:
            data: Dictionary to serialize as JSON
        """
        with open(self.file_path, "w") as f:
            json.dump(data, f, indent=2)
