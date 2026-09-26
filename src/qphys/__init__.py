"""qphys -- quantum formalism as a modeling language."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("qphys")
except PackageNotFoundError:  # a source tree that was never installed
    __version__ = "0+unknown"
