import logging
import os
import sys

logger = logging.getLogger()

class EventLogger:
    logger = logging.getLogger()
    pre_logger_messages = []  # Temporary storage for messages before logger is configured

    def __init__(self, _log_directory):
        # Set the logging level
        self.logger.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        # console_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)-8s] [%(module)-14s] [%(funcName)-14s] [%(threadName)-14s] %(msg)s"))
        console_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)-6s] [%(module)-18s] [%(funcName)-20s] %(msg)s"))

        log_filename =os.path.join(_log_directory, os.path.basename(_log_directory)) +'.log'
        
        # Create a file handler
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.DEBUG)
        # file_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)-8s] [%(module)-14s] [%(funcName)-14s] [%(threadName)-14s] %(msg)s"))
        file_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)-6s] [%(module)-18s] [%(funcName)-20s] %(msg)s"))

        logger.handlers.clear()
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        # Log the messages stored before the logger was configured
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