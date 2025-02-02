import datetime
import logging

import os

import time

LOG_DIR = os.path.join(os.path.dirname(__file__), "../../logs")
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# filter_kw = []
filter_kw = ["Prun", "Handle", "start at", "end at", "pend/submit at", "failed placed", "#"]


class LogFilter(logging.Filter):
    def filter(self, record):
        for kw in filter_kw:
            if kw in record.msg:
                return False
        return True


# set up a custom logger
def get_logger(name='logger', level='INFO', mode='w', fh=True, ch=True, prefix=""):
    log = logging.getLogger(name)

    t = time.time()
    s = datetime.datetime.fromtimestamp(t).strftime('%m-%d_%H-%M-%S')
    fh = logging.FileHandler(os.path.join(LOG_DIR, f'log_{s}.txt'), mode) if fh else None

    ch = logging.StreamHandler() if ch else None
    # ch.addFilter(LogFilter())

    if level == "INFO":
        log.setLevel(logging.INFO)
        fh.setLevel(logging.INFO)
        ch.setLevel(logging.INFO)
    elif level == "DEBUG":
        log.setLevel(logging.DEBUG)
        fh.setLevel(logging.DEBUG)
        ch.setLevel(logging.DEBUG)
    elif level == "ERROR":
        log.setLevel(logging.ERROR)
        fh.setLevel(logging.ERROR)
        ch.setLevel(logging.ERROR)

    formatter = logging.Formatter('[%(asctime)s] [%(filename)s:%(lineno)s] [%(levelname)s]: %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    if fh:
        log.addHandler(fh)
    if ch:
        log.addHandler(ch)

    return log


logger = get_logger()
