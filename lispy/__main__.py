"""Mini interpreter alternative to reparsed hell."""
import getopt
import platform
from tokenize import TokenError
import traceback
import sys
from collections.abc import Generator

import lispy
from lispy import Context, lispy_exec, lispy_parse

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


def _fake_readline(src: bytes) -> Generator[bytes, None, None]:
    yield from src.splitlines(keepends=True)
    yield bytes()


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
            for i in lispy_exec(lispy_parse(file.readline)):
                if i is not None:
                    print(i)
            sys.exit()

    if not sys.stdin.isatty():
        lispy_exec(lispy_parse(sys.stdin.buffer.readline))
    else:
        try:
            import readline
            readline.parse_and_bind("")
        except ModuleNotFoundError:
            pass

        print(ENVIRONMENT)
        print(COPYRIGHT)

        ctx = Context()

        while True:
            try:
                src = input("(>>>) ").encode() + b"\n"
                while True:
                    try:
                        node = lispy_parse(_fake_readline(src).__next__)
                    except TokenError:
                        src += input("(...) ").encode() + b"\n"
                        continue
                    break
            except KeyboardInterrupt:
                print("\nKeyboardInterrupt")
                continue
            except EOFError:
                print()
                break

            if src == b"help\n":
                print("Available symbols:")
                print(" ", ", ".join(lispy.stdlib.keys()))
                print("Press EOF (Ctrl+D) to exit. Ctrl+C stops current code.")
                continue

            try:
                for i in lispy_exec(node, ctx):
                    if i is not None:
                        print(i)
            except (Exception, KeyboardInterrupt):
                traceback.print_exc()


if __name__ == "__main__":
    main()
