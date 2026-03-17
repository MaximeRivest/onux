from __future__ import annotations

import doctest
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import onux.examples as examples
import onux.signatures as signatures


def load_tests(loader: unittest.TestLoader, tests: unittest.TestSuite, ignore: str | None) -> unittest.TestSuite:
    optionflags = doctest.ELLIPSIS
    tests.addTests(doctest.DocTestSuite(examples, optionflags=optionflags))
    tests.addTests(doctest.DocTestSuite(signatures, optionflags=optionflags))
    return tests
