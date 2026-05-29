"""
Save and load test results in JSON and CSV formats.
"""

import json
import os
import numpy as np
from datetime import datetime


def _convert_to_serializable(obj):
    """Recursively convert numpy types to Python native types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, complex):
        return {'real': obj.real, 'imag': obj.imag}
    if isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_to_serializable(item) for item in obj]
    return obj


def save_json(results: dict, filepath: str) -> str:
    """Save results dict to JSON file."""
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    serializable = _convert_to_serializable(results)
    with open(filepath, 'w') as f:
        json.dump(serializable, f, indent=2)
    return filepath


def save_csv(rows: list, columns: list, filepath: str) -> str:
    """Save tabular data to CSV file."""
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(','.join(columns) + '\n')
        for row in rows:
            f.write(','.join(str(v) for v in row) + '\n')
    return filepath


def save_results(results: dict, filepath: str = None) -> str:
    """
    Save results to a timestamped JSON file if no filepath given.
    """
    if filepath is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = f'data/results_{timestamp}.json'
    return save_json(results, filepath)