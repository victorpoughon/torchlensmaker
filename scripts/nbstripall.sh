#!/usr/bin/env bash

set -euo pipefail

# Find and process all *.ipynb files in the specified directories and their subdirectories
find src/ docs/src/ test_notebooks/ -type f -name "*.ipynb" -exec nbstripout-fast {} +
