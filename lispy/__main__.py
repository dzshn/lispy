import platform
import traceback
import sys

import lispy
from lispy import lispy_exec


if __name__ == "__main__":
    if not sys.stdin.isatty():
        lispy_exec(lambda: sys.stdin.readline().encode())
    else:
        try:
            import readline
            readline.parse_and_bind("")
        except ModuleNotFoundError:
            pass

        print(
            f"LisPy {lispy.__version__} on "
            f"{platform.python_implementation()} {platform.python_version()} "
            f"on {platform.system()}"
        )
        print("Copyright (c) 2022 Sofia Lima")

        while True:
            try:
                line = input("(>>>) ").encode()
            except KeyboardInterrupt:
                print("\nKeyboardInterrupt")
                continue
            except EOFError:
                print()
                break

            if line == b"help":
                print("Available symbols:")
                print(" ", ", ".join(lispy.stdlib.keys()))
                print("Press EOF (Ctrl+D) to exit. Ctrl+C stops current code.")
                continue

            try:
                for i in lispy_exec((lambda: (yield from [line]))().__next__):
                    if i is not None:
                        print(i)
            except (Exception, KeyboardInterrupt):
                traceback.print_exc()
