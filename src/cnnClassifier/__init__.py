import os
import sys
import logging

logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"   # Defines the logging format with timestamp, log level, module name, and message

log_dir = "logs"                                              # Sets the directory where log files will be stored
log_filepath = os.path.join(log_dir,"running_logs.log")       # Defines the file path for the log file within the log directory
os.makedirs(log_dir, exist_ok=True)                           # Creates the logs directory if it does not already exist


logging.basicConfig(
    level = logging.INFO,
    format = logging_str,
    handlers = [
        logging.FileHandler(log_filepath),  # Handler for writing log messages to the specified file
        logging.StreamHandler(sys.stdout)   # Handler for displaying log messages in the console (stdout)
    ]
)

logger = logging.getLogger("cnnClassifierLogger")       # Creates a logger with the name "cnnClassifierLogger" for logging messages specific to this module
