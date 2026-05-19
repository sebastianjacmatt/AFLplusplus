from .config import Config
from .mutator import Mutator

_m = Mutator(Config.load(), collection="interesting", trainer="covrl")


def init(seed: int) -> None:
    _m.init(seed)


def splice_optout() -> bool:
    return True


def fuzz_count(buf: bytearray) -> int:
    return _m.fuzz_count(buf)


def fuzz(buf: bytearray, add_buf: bytearray, max_size: int) -> bytearray:
    return _m.fuzz(buf, add_buf, max_size)


def post_run() -> None:
    _m.post_run()


def queue_new_entry(filename_new_queue: str, filename_orig_queue: str) -> None:
    _m.queue_new_entry(filename_new_queue, filename_orig_queue)


def queue_get(filename: str) -> bool:
    return _m.queue_get(filename)


def deinit() -> None:
    _m.deinit()