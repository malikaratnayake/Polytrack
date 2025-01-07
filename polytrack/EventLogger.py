import logging
import os

logger = logging.getLogger()

class EventLogger:
    logger = logging.getLogger()

    def __init__(self, _log_directory):
        # Set the logging level
        self.logger.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        # console_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)-8s] [%(module)-14s] [%(funcName)-14s] [%(threadName)-14s] %(msg)s"))
        console_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)-8s] [%(module)-14s] [%(funcName)-14s] [%(threadName)-14s] %(msg)s"))

        log_filename =os.path.join(_log_directory, os.path.basename(_log_directory)) +'_log.log'
        print(log_filename)
        
        # Create a file handler
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.DEBUG)
        # file_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)-8s] [%(module)-14s] [%(funcName)-14s] [%(threadName)-14s] %(msg)s"))
        file_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)-8s] [%(module)-14s] [%(funcName)-14s] [%(threadName)-14s] %(msg)s"))

        logger.handlers.clear()
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        return None
    
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