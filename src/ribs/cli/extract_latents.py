import sys

from . import main

if __name__ == "__main__":
    raise SystemExit(main(["extract-latents", *sys.argv[1:]]))
