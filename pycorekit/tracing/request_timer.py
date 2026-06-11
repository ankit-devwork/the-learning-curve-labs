import time
from contextlib import contextmanager

@contextmanager
def request_timer():
    start = time.time()
    yield lambda: round((time.time() - start) * 1000, 2)
