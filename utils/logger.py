import logging
import sys
import torch
import torch.distributed as dist
import os

DEBUG = False

get_trace = getattr(sys, 'gettrace', None)
if get_trace():
    print('Program runs in Debug mode')
    DEBUG = True


class RankFilter(logging.Filter):

    def filter(self, record):
        if not dist.is_available() or not dist.is_initialized():
            # No distributed setup, allow all log records
            return True
        else:
            # Only allow log records to pass through if the rank is 0
            return int(os.environ['RANK']) == 0


def get_logger(name: str):
    logger = logging.getLogger(name=name)
    logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)

    formatter = logging.Formatter(fmt='%(levelname)s: [%(asctime)s]  %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG if DEBUG else logging.INFO)
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(RankFilter())
    logger.addHandler(stream_handler)

    return logger
