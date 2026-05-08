"""
event_logger.py
===============
Logging initialisation for Polytrack.

EventLogger sets up the root logger with:
  - A *console handler* whose level is set above CRITICAL (i.e. silent) so
    that Polytrack's progress output is not mixed with log lines.
  - A *file handler* that writes to <output_dir>/<output_dir>.log at the
    level specified by ``output.log_level`` in the YAML config.

The class-level ``pre_logger_messages`` list acts as a temporary buffer for
calls made before the output directory exists (e.g. during argument parsing).
These are flushed to the configured handlers on the first ``__init__`` call.
"""

import logging
import os

# Module-level root logger reference used by the flush helper.
logger = logging.getLogger()


class EventLogger:
    """
    Configures the root Python logger for a Polytrack processing run.

    Args:
        _log_directory: Directory where the ``.log`` file will be written.
            The log filename is ``<basename(_log_directory)>.log``.
        log_level: String level name (``"DEBUG"``, ``"INFO"``, ``"WARNING"``,
            ``"ERROR"``).  Defaults to ``"INFO"``.  Both the root logger and
            the file handler are set to this level so that DEBUG output is
            captured when requested.
    """

    logger = logging.getLogger()
    pre_logger_messages: list = []  # Buffer for messages logged before init

    _LOG_FORMAT = "[%(asctime)s] [%(levelname)-6s] [%(module)-18s] [%(funcName)-20s] %(msg)s"

    def __init__(self, _log_directory, log_level: str = "INFO") -> None:
        log_level_value = logging._nameToLevel.get(str(log_level).upper(), logging.INFO)

        self.logger.setLevel(log_level_value)

        # Console handler: intentionally silenced (progress is printed directly
        # to stdout via print()), but kept in the handler chain so external
        # tooling that reads stderr/stdout gets nothing garbled.
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.CRITICAL + 1)
        console_handler.setFormatter(logging.Formatter(self._LOG_FORMAT))

        log_filename = os.path.join(_log_directory, os.path.basename(_log_directory)) + ".log"

        # File handler: write at the user-configured level (not hardcoded INFO).
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(log_level_value)
        file_handler.setFormatter(logging.Formatter(self._LOG_FORMAT))

        logger.handlers.clear()
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        # Replay any messages buffered before the logger was configured.
        self._flush_pre_logger()

        return None
    
    def _flush_pre_logger(self):
        """Log messages stored before the logger was configured."""
        for level, msg, args, kwargs in self.pre_logger_messages:
            getattr(self.logger, level)(msg, *args, **kwargs)
        self.pre_logger_messages.clear()

    @classmethod
    def temp_log(cls, level, msg, *args, **kwargs):
        """Temporarily store log messages before the logger is configured."""
        cls.pre_logger_messages.append((level, msg, args, kwargs))
    
    def debug(self, msg, *args, **kwargs):
        """Log a debug message."""
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        """Log an info message."""
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        """Log a warning message."""
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        """Log an error message."""
        self.logger.error(msg, *args, **kwargs)
