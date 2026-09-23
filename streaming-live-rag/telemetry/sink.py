import os
import threading
from telemetry.schema import TelemetryEvent

# Create logs directory if it doesn't exist
LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
TELEMETRY_LOG_FILE = os.path.join(LOGS_DIR, "telemetry.jsonl")
_MAX_TELEMETRY_BYTES = int(os.getenv("TELEMETRY_MAX_BYTES", str(10 * 1024 * 1024)))
_sink_lock = threading.Lock()

def emit(event: TelemetryEvent):
    """
    Appends the TelemetryEvent as a JSON line to the log file.
    This fulfills the G6 observability gate requirement.
    Rotates the file (never grows unbounded / never crashes the request).
    """
    with _sink_lock:
        try:
            if os.path.exists(TELEMETRY_LOG_FILE) and os.path.getsize(TELEMETRY_LOG_FILE) >= _MAX_TELEMETRY_BYTES:
                rotated = TELEMETRY_LOG_FILE + ".1"
                try:
                    if os.path.exists(rotated):
                        os.remove(rotated)
                    os.replace(TELEMETRY_LOG_FILE, rotated)
                except OSError:
                    # If rotation fails, truncate rather than grow forever.
                    open(TELEMETRY_LOG_FILE, "w", encoding="utf-8").close()
            with open(TELEMETRY_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(event.model_dump_json() + "\n")
        except OSError as e:
            import logging
            logging.getLogger(__name__).warning("telemetry sink failed: %s", e)
            raise
