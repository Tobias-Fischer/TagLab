"""
Entry point `taglab` of the conda package: runs TagLab.py from the folder where the package installed the TagLab tree
(<prefix>/share/taglab), or from the folder of this file when run from a source checkout.
"""

import os
import runpy
import sys


def taglab_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(here, "TagLab.py")):
        return here
    return os.path.join(sys.prefix, "share", "taglab")


def main():
    path = taglab_dir()
    sys.path.insert(0, path)    # TagLab imports its own `source` and `models` packages
    sys.argv[0] = os.path.join(path, "TagLab.py")
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
