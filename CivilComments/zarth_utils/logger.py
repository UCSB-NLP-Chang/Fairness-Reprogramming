import sys
import logging
from zarth_utils.general_utils import get_random_time_stamp, makedir_if_not_exist


def get_logger(path_log="%s.log" % get_random_time_stamp()):
    """
    Set up the logger. Note that the setting will also impact the default logging logger, which means that simply
    using logging.info() will output the logs to both stdout and the filename_log.
    :param path_log: the filename of the log
    :type path_log: str
    """
    ret_logger = logging.getLogger()
    ret_logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s: %(message)s', datefmt='%Y-%m-%d-%H:%M:%S')

    if not ret_logger.handlers:
        path_log = "%s.log" % path_log if not path_log.endswith(".log") else path_log
        fh = logging.FileHandler(path_log)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)

        ret_logger.addHandler(ch)
        ret_logger.addHandler(fh)

    return ret_logger
