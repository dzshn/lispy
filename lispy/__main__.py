import traceback
from lispy import lispy_exec


if __name__ == "__main__":
    try:
        import readline
        readline.parse_and_bind("")
    except ModuleNotFoundError:
        pass

    while True:
        try:
            line = input("% ").encode()
        except EOFError:
            print()
            break

        try:
            lispy_exec((lambda: (yield from [line]))().__next__)
        except Exception:
            traceback.print_exc()
