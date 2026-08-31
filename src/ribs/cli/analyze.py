import sys

from . import main

if __name__ == "__main__":
    raise SystemExit(main(["analyze", *sys.argv[1:]]))
