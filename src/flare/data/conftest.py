# -*- coding: UTF-8 -*-
import os
import sys

# Add project root to sys.path for absolute imports
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
