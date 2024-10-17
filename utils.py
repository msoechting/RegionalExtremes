import datetime
import os
import time
import sys


def int_or_none(value):
    if value.lower() == "none":
        return None
    return int(value)


class Logger:
    def __init__(self, filepath: str = "log.txt", write_to_file: bool = True):
        self.write_to_file = write_to_file

        if not self.write_to_file:
            return
        
        self.filepath = filepath / "log.txt"

        # Ensure the log file exists
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.filepath.touch(exist_ok=True)


    def printt(self, message: str):
        """
        Small function to track the different steps of the program with the time.
        Prints to console and appends to a file.
        """
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{current_time}] {message}"

        # Print to console
        print(formatted_message)

        # Append to file
        if self.write_to_file:
            with open(self.filepath, "a") as file:
                file.write(formatted_message + "\n")

    def raiset(self, error):
        """
        Small function to track the different steps of the program with the time.
        Prints to console and appends to a file.
        """
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{current_time}] ValueError: {error}"

        # Print to console
        print(formatted_message)

        if self.write_to_file:
            # Append to file
            with open(self.filepath, "a") as file:
                file.write(formatted_message + "\n")

        raise error


# Global variable to hold the logger instance
logger = None


def initialize_logger(filepath: str = "log.txt", write_to_file: bool = True):
    global logger
    logger = Logger(filepath, write_to_file)


def printt(message: str):
    if logger is None:
        raise RuntimeError("Logger not initialized. Call initialize_logger() first.")
    logger.printt(message)


def raiset(message):
    if logger is None:
        raise RuntimeError("Logger not initialized. Call initialize_logger() first.")
    logger.raiset(message)


# Not yet used
# def timing_decorator(func):
#     def wrapper(*args, **kwargs):
#         start_time = time.time()
#         result = func(*args, **kwargs)
#         end_time = time.time()
#         printt(f"Function {func.__name__} took {end_time - start_time:.4f} seconds")
#         return result
#
#     return wrapper
