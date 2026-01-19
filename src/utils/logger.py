'''
A Logger module for logging.

This module provides a Logger class to handle logging messages to a file
and optionally to the console. The logging levels supported include
`debug`, `info`, `warning`, `error`, and `critical` (case-insensitive).

Classes:
    Logger: Logs information to logfile and console at desired levels.

Functions:
    close_all_logs(): Closes all active log files in the project.

Example usage:

    # import required modules
    import datetime
    import logging
    import utils_logger.Logger

    # set up the Logger instance
    logger = utils_logger.Logger(name='proj_logger',
                                 log_file='./log/main.log',
                                 log_lvl=logging.DEBUG,
                                 console_lvl=logging.WARNING)

    # log messages with varying log levels
    logger.log('info', 'This is an info message')
    logger.log('warning', 'This is a warning message')
    logger.log('error', 'This is an error message')

    # close the logger and rename the log file
    logger.close()
    finish_time = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    os.rename('mainlog', f'mainlog_{finish_time}.log')

    # example lines in the resulting log file
    """
    2025-03-14 03:14:15,926-proj_logger-INFO - 'This is an info message'
    """
'''

# standard imports
# import datetime
import logging
import os

class Logger():
    '''
    A class to handle logging messages to a file and optionally to the
    console.

    Attributes:
        logger (logging.Logger): The logger instance.
    '''

    def __init__(
            self,
            name: str | None=None,
            log_file: str | None=None,
            log_lvl: int=logging.DEBUG,
            console_lvl: int | None=logging.INFO
        ):
        '''
        Initializes the Logger instance.

        If `name` is not provided, the script file name with be used and
        if `log_file` is not provided, a proj.log file will be created
        at the current working directory.

        Args:
            name (str, optional): Name of the logger. If None use the
                script name.
            log_file (str, optional): Path to the log file.
            log_lvl (int, optional): Logging level for the file handler.
            console_lvl (int, optional): Logging level for the console
                handler. If None, console logging is disabled.
        '''

        # gather arguments
        if name is None:
            name = os.path.basename(__file__)
        if log_file is None:
            default_dirpath = f'{os.getcwd()}/logs'
            os.makedirs(default_dirpath, exist_ok=True)
            # timestap = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = f'{default_dirpath}/proj.log' # default log file

        # assign attributes for potential access
        self.name = name
        self.log_file = log_file

        # init logger attribute
        self.logger = logging.getLogger(name)
        # set log level accordingly
        self.logger.setLevel(log_lvl)
        # prevent log messages from propagating to the root logger
        self.logger.propagate = False

        # check if the logger already has handlers to avoid duplication
        if not self.logger.hasHandlers():

            # create a file handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_lvl)
            # create a formatter and set it for the file handler
            formatter = logging.Formatter(
                '%(asctime)s-%(name)s-%(levelname)s\t- %(message)s')
            file_handler.setFormatter(formatter)
            # add the file handler to the logger
            self.logger.addHandler(file_handler)

            # add a console handler if chosen to
            if console_lvl is not None:
                console_handler = logging.StreamHandler()
                console_handler.setLevel(console_lvl)
                console_handler.setFormatter(formatter)
                self.logger.addHandler(console_handler)


    def log(self, level: str, message: str, skip_log: bool=False) -> None:
        '''
        Logs a message with the specified logging level.

        Args:
            level (str): The logging level includes `'debug'`, `'info'`,
                `'warning'`, `'error'`, and `'critical'`.
            message (str): The message to log.
            skip_log (bool, optional): Flag whether to log or not.
        '''

        # skip logging if chooses so
        if skip_log:
            return

        # define log levels
        log_levels = {
            'debug': self.logger.debug,
            'info': self.logger.info,
            'warning': self.logger.warning,
            'error': self.logger.error,
            'critical': self.logger.critical
        }

        # log accordingly and defaulting to 'info' if level is unrecognized
        # case-insensitive
        log_method = log_levels.get(level.lower(), self.logger.info)
        log_method(message)

    def close(self) -> None:
        '''Closes the file handler.'''

        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)

def log_separator(log: Logger, sep: str='=', ln: int=90) -> None:
    '''
    Log a separator with a repeated string and length.

    Args:
        log (Logger): Logs to file and console.
        sep (str, optional): Repeats to form a separator line
            (default: `'='`).
        ln (int, optional): Length of the line (default: 90).
    '''

    log.log('INFO', sep * ln)
