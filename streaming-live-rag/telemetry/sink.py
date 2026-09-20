import os
import json
from telemetry.schema import TelemetryEvent

# Create logs directory if it doesn't exist
LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
TELEMETRY_LOG_FILE = os.path.join(LOGS_DIR, "telemetry.jsonl")

def emit(event: TelemetryEvent):
    """
    Appends the TelemetryEvent as a JSON line to the log file.
    This fulfills the G6 observability gate requirement.
    """
    with open(TELEMETRY_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(event.model_dump_json() + "\n")
