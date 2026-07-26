import logging
import re
import threading
from collections import deque
from datetime import datetime, timezone


ANSI_ESCAPE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


class RuntimeLogBroker:
    def __init__(self, capacity=1000):
        self._entries = deque(maxlen=capacity)
        self._sequence = 0
        self._condition = threading.Condition()

    def publish(self, message, level="info", source="server", job_id=None):
        clean_message = ANSI_ESCAPE.sub("", str(message)).replace("\r", "").strip()
        if not clean_message:
            return None

        with self._condition:
            self._sequence += 1
            entry = {
                "id": self._sequence,
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "level": str(level).lower(),
                "source": str(source),
                "job_id": job_id,
                "message": clean_message,
            }
            self._entries.append(entry)
            self._condition.notify_all()
            return entry

    def recent(self, limit=250):
        safe_limit = max(1, min(int(limit), self._entries.maxlen))
        with self._condition:
            return list(self._entries)[-safe_limit:]

    def wait_after(self, sequence, timeout=15):
        with self._condition:
            if not self._entries or self._entries[-1]["id"] <= sequence:
                self._condition.wait(timeout=timeout)
            return [entry for entry in self._entries if entry["id"] > sequence]


runtime_logs = RuntimeLogBroker()


def publish_log(message, level="info", source="server", job_id=None):
    return runtime_logs.publish(message, level=level, source=source, job_id=job_id)


class RuntimeLogHandler(logging.Handler):
    def emit(self, record):
        try:
            publish_log(
                self.format(record),
                level=record.levelname.lower(),
                source=record.name,
                job_id=getattr(record, "job_id", None),
            )
        except Exception:
            self.handleError(record)
