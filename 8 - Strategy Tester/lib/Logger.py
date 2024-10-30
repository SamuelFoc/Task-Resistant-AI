import logging
from logging.handlers import RotatingFileHandler
import os

class Logger:
    _instance = None  # Class variable to store the singleton instance
    _test_name = None  # Class variable to store the test name

    def __new__(cls, test_name=None):
        """
        Controls the creation of a single Logger instance.

        Parameters:
        - test_name (str): Name of the test, used to create a unique log directory.

        Returns:
        - Instance of Logger.
        """
        if cls._instance is None:
            # First-time creation of the instance
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._initialized = False  # Prevent multiple initializations

            # Set test_name only if provided during the first initialization
            if test_name is None:
                raise ValueError("test_name must be provided for the first initialization of Logger.")
            cls._test_name = test_name
        elif test_name and test_name != cls._test_name:
            raise ValueError("Logger is already initialized with a different test_name.")
        return cls._instance

    def __init__(self, test_name=None):
        """
        Initializes the Logger class with a specified test name.

        Parameters:
        - test_name (str): Name of the test, used to create a unique log directory.
        """
        if not self._initialized:  # Initialize only once
            self.log_dir = os.path.join("tests", self._test_name)
            os.makedirs(self.log_dir, exist_ok=True)
            self.log_file_path = os.path.join(self.log_dir, "strategy_test.log")
            self._initialized = True  # Mark as initialized

    def get_logger(self, name):
        """
        Returns a logger configured to log messages for the specified test.

        Parameters:
        - name (str): Name of the logger (typically the class name).

        Returns:
        - logger (logging.Logger): Configured logger instance.
        """
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        # Check if handlers already exist to avoid duplicate logs
        if not logger.handlers:
            file_handler = RotatingFileHandler(self.log_file_path, maxBytes=10**6, backupCount=5)
            stream_handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
            
            file_handler.setFormatter(formatter)
            stream_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
            logger.addHandler(stream_handler)

        return logger
