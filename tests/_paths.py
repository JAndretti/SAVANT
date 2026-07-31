#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared path resolution for the verification scripts.

These scripts live in tests/ but drive the solver binary at the repository root
and import validate.py from tools/. Both locations are derived from __file__ so
the scripts work whatever the current working directory is.

Importing this module has the side effect of putting tools/ on sys.path, which
is what makes `from validate import read_bundle` work from tests/.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
BIN = os.path.join(ROOT, "cw")
BIN_DBG = os.path.join(ROOT, "cw_dbg")
DATA = os.path.join(ROOT, "data")
EDGE = os.path.join(ROOT, "edge")
VALIDATE = os.path.join(TOOLS, "validate.py")

if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)
