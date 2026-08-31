import sys

from . import main

if __name__ == "__main__":
    raise SystemExit(main(["prepare-data", *sys.argv[1:]]))
