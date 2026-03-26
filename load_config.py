import os
import json
import sys

# Safely resolve path relative to this file
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

try:
    with open(CONFIG_PATH, "r") as f:
        CFG = json.load(f)
except FileNotFoundError:
    print(f"\n[FATAL] Could not find configuration file at:\n  {CONFIG_PATH}")
    sys.exit(1)

# Centralized constants derived from config
DECIMATE_FACTOR = CFG["hardware"]["decimate_factor"]
ENABLE_INTEL_IGPU = CFG["hardware"]["enable_intel_igpu"]
DEPTH_SCALE = 1000.0
