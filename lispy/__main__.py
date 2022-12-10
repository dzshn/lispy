"""Mini interpreter alternative to reparsed hell."""
import getopt
import platform
import traceback
import sys
from collections.abc import Generator

import lispy
from lispy import lispy_exec

ENVIRONMENT = (
    f"LisPy {lispy.__version__} on "
    f"{platform.python_implementation()} {platform.python_version()} "
    f"on {platform.system()}"
)
COPYRIGHT = "Copyright (c) 2022 Sofia Lima <me@dzshn.xyz>"
USAGE = f"""\
LisPy {lispy.__version__}
{COPYRIGHT}

LisPy is a cursed Lisp dialect using Python syntax.

USAGE
    lispy [script.lpy] ...

SEE ALSO
    Project url  https://github.com/dzshn/lispy
"""


def _fake_readline(line: bytes) -> Generator[bytes, None, None]:
    yield from [line]


def main() -> None:
    try:
        opts, args = getopt.getopt(sys.argv[1:], "h", ["help", "version"])
    except getopt.GetoptError as exc:
        print(USAGE)
        print()
        sys.exit(str(exc))
    for i, _ in opts:
        if i in {"-h", "--help"}:
            print(USAGE)
            sys.exit()
        if i == "--version":
            print(ENVIRONMENT)
            sys.exit()

    if args:
        with open(args[0], "rb") as file:
            lispy_exec(file.readline)
            sys.exit()

    if not sys.stdin.isatty():
        lispy_exec(lambda: sys.stdin.readline().encode())
    else:
        try:
            import readline
            readline.parse_and_bind("")
        except ModuleNotFoundError:
            pass

        print(ENVIRONMENT)
        print(COPYRIGHT)

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
                # TODO: multiline input
                for i in lispy_exec(_fake_readline(line).__next__):
                    if i is not None:
                        print(i)
            except (Exception, KeyboardInterrupt):
                traceback.print_exc()


if __name__ == "__main__":
    main()
