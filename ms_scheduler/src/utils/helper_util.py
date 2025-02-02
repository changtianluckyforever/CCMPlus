import time
from contextlib import contextmanager
from functools import wraps

from tabulate import tabulate


def tabulate_print(data_frame, rows=5, headers="keys", tail=False):
    if tail:
        print(tabulate(data_frame.tail(rows), headers=headers, tablefmt='psql'))
    else:
        print(tabulate(data_frame.head(rows), headers=headers, tablefmt='psql'))


def time_str(s):
    """
    Convert seconds to a nicer string showing days, hours, minutes and seconds
    """
    days, remainder = divmod(s, 60 * 60 * 24)
    hours, remainder = divmod(remainder, 60 * 60)
    minutes, seconds = divmod(remainder, 60)
    string = ""
    if days > 0:
        string += "{:d} days, ".format(int(days))
    if hours > 0:
        string += "{:d} hours, ".format(int(hours))
    if minutes > 0:
        string += "{:d} minutes, ".format(int(minutes))
    string += "{:d} seconds".format(int(seconds))
    return string


@contextmanager
def time_block(label, enable=True):
    if not enable:
        yield
    else:
        start = time.perf_counter()  # time.process_time()
        try:
            yield
        finally:
            end = time.perf_counter()
            print(f"{label} : {time_str(end - start)}")


def timethis(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        r = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__module__}.{func.__name__} : {end - start}")
        return r

    return wrapper
