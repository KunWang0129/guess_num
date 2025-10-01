import csv
from pathlib import Path
from typing import Dict, Optional


class CSVLogger:
    """CSV logger for training metrics."""

    def __init__(self, filename: str, fieldnames: list, log_dir: Optional[str] = None):
        """Initialize CSV logger.

        Args:
            filename: CSV filename
            fieldnames: List of column names
            log_dir: Directory to save the CSV file (optional, can be set later)
        """
        self.filename = filename
        self.fieldnames = fieldnames
        self.log_dir = Path(log_dir) if log_dir else None
        self.file = None
        self.writer = None

        if self.log_dir:
            self._initialize_file()

    def _initialize_file(self):
        """Initialize the CSV file."""
        if self.file is not None:
            return  # Already initialized

        self.filepath = self.log_dir / self.filename

        # Create parent directory if it doesn't exist
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        # Open file and write header
        self.file = open(self.filepath, 'w', newline='')
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
        self.writer.writeheader()
        self.file.flush()

    def set_log_dir(self, log_dir: str):
        """Set the log directory and initialize the file.

        Args:
            log_dir: Directory to save the CSV file
        """
        self.log_dir = Path(log_dir)
        self._initialize_file()

    def log(self, metrics: Dict):
        """Log metrics to CSV.

        Args:
            metrics: Dictionary of metrics to log
        """
        if self.writer is None:
            raise RuntimeError("CSV logger is closed")

        # Only write fields that are in fieldnames
        filtered_metrics = {k: v for k, v in metrics.items() if k in self.fieldnames}
        self.writer.writerow(filtered_metrics)
        self.file.flush()

    def close(self):
        """Close the CSV file."""
        if self.file is not None:
            self.file.close()
            self.file = None
            self.writer = None

    def __del__(self):
        """Cleanup on deletion."""
        self.close()
