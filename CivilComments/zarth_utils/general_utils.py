import os
import random
import datetime


def get_random_time_stamp():
    """
    Return a random time stamp.
    :return: random time stamp
    :rtype: str
    """
    return "%d-%s" % (random.randint(100, 999), datetime.datetime.now().strftime('%Y.%m.%d-%H.%M.%S'))


def makedir_if_not_exist(name):
    """
    Make the directory if it does not exist.
    :param name: dir name
    :type name: str
    """
    if not os.path.exists(name):
        os.makedirs(name)
