"""A memory watchdog for benchmarks: exit before a runaway allocation reaches swap."""

import os
import sys
import threading
import time

import lucid


def guard(limit_gb: float = 20.0) -> None:
    def watch() -> None:
        while True:
            used = lucid.metal.memory_allocated() / 1e9
            if used > limit_gb:
                sys.stdout.write(
                    f"\nGUARD: {used:.1f} GB in use > {limit_gb} GB, aborting\n"
                )
                sys.stdout.flush()
                os._exit(3)
            time.sleep(0.1)

    threading.Thread(target=watch, daemon=True).start()
