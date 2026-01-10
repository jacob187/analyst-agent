"""
Unit tests for LocalLogger class.
"""

import pytest
import json
import os
from pathlib import Path
from database.local_logger import LocalLogger


class TestLocalLogger:
    """Test cases for LocalLogger persistence layer."""

    def test_initialization(self, tmp_path, monkeypatch):
        """Test LocalLogger initializes with correct file path."""
        # Change the root directory to tmp_path for testing
        monkeypatch.setattr(
            "database.local_logger.Path",
            lambda x: tmp_path if "local_logger.py" in str(x) else Path(x),
        )

        logger = LocalLogger()

        # Verify the data directory was created
        assert (tmp_path / "data").exists()

    def test_read_json_nonexistent_file(self, tmp_path, monkeypatch):
        """Test reading from nonexistent file creates empty file."""
        test_file = tmp_path / "data" / "test_data.json"

        # Mock the file_path
        logger = LocalLogger()
        logger.file_path = test_file

        result = logger.read_json()

        assert result == {}
        assert test_file.exists()

    def test_read_json_existing_file(self, tmp_path):
        """Test reading from existing JSON file."""
        # Create test data
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        test_file = data_dir / "test_data.json"

        test_data = {"AAPL": {"price": 150.0}, "GOOGL": {"price": 2800.0}}
        with open(test_file, "w") as f:
            json.dump(test_data, f)

        # Test reading
        logger = LocalLogger()
        logger.file_path = test_file
        result = logger.read_json()

        assert result == test_data
        assert "AAPL" in result
        assert result["AAPL"]["price"] == 150.0

    def test_read_json_malformed_file(self, tmp_path):
        """Test reading malformed JSON returns empty dict."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        test_file = data_dir / "test_data.json"

        # Write malformed JSON
        with open(test_file, "w") as f:
            f.write("{invalid json content")

        logger = LocalLogger()
        logger.file_path = test_file
        result = logger.read_json()

        assert result == {}

    def test_write_json_success(self, tmp_path):
        """Test writing data to JSON file."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        test_file = data_dir / "test_data.json"

        test_data = {
            "AAPL": {
                "sec_data": {"risk_factors": "Sample risks"},
                "technical_data": {"price": 150.0},
            }
        }

        logger = LocalLogger()
        logger.file_path = test_file
        logger.write_json(test_data)

        # Verify file was written
        assert test_file.exists()

        # Verify content
        with open(test_file, "r") as f:
            written_data = json.load(f)

        assert written_data == test_data

    def test_write_json_overwrites_existing(self, tmp_path):
        """Test writing overwrites existing file content."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        test_file = data_dir / "test_data.json"

        # Write initial data
        initial_data = {"AAPL": {"price": 150.0}}
        with open(test_file, "w") as f:
            json.dump(initial_data, f)

        # Write new data
        new_data = {"GOOGL": {"price": 2800.0}}
        logger = LocalLogger()
        logger.file_path = test_file
        logger.write_json(new_data)

        # Verify new data overwrote old data
        with open(test_file, "r") as f:
            written_data = json.load(f)

        assert written_data == new_data
        assert "AAPL" not in written_data
        assert "GOOGL" in written_data

    def test_write_json_creates_directory(self, tmp_path):
        """Test write_json creates directory if it doesn't exist."""
        test_file = tmp_path / "new_data" / "test_data.json"

        test_data = {"AAPL": {"price": 150.0}}

        logger = LocalLogger()
        logger.file_path = test_file
        logger.write_json(test_data)

        # Verify directory and file were created
        assert test_file.parent.exists()
        assert test_file.exists()

    def test_round_trip_read_write(self, tmp_path):
        """Test round-trip write and read operation."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        test_file = data_dir / "test_data.json"

        original_data = {
            "AAPL": {
                "sec_data": {
                    "risk_factors": "Business risks include...",
                    "mda": "Management discussion...",
                },
                "technical_data": {
                    "price": 150.0,
                    "moving_averages": {"ma_50": 148.0, "ma_200": 145.0},
                },
            },
            "GOOGL": {
                "sec_data": {"risk_factors": "Technology risks..."},
                "technical_data": {"price": 2800.0},
            },
        }

        logger = LocalLogger()
        logger.file_path = test_file

        # Write data
        logger.write_json(original_data)

        # Read data back
        read_data = logger.read_json()

        assert read_data == original_data
        assert read_data["AAPL"]["technical_data"]["price"] == 150.0
        assert read_data["GOOGL"]["technical_data"]["price"] == 2800.0
