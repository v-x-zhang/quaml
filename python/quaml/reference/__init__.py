"""Pure-NumPy reference implementations.

These define the semantics the C++ engine must reproduce. They favour clarity
over speed and are the oracle for differential tests.
"""

from . import features, quote, rolling, signing

__all__ = ["features", "quote", "rolling", "signing"]
