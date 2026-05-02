import os
import json

def log_to_file(filepath, log_entry):
    """
    Safe JSONL logger (no corruption, no path errors)
    """

    # -----------------------------
    # FIX: HANDLE FILEPATH PROPERLY
    # -----------------------------
    directory = os.path.dirname(filepath)

    if directory != "":
        os.makedirs(directory, exist_ok=True)

    # -----------------------------
    # WRITE AS JSONL
    # -----------------------------
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")