#!/usr/bin/env bash

set -euo pipefail

# nbstripout-fast examples/*.ipynb test_notebooks/*.ipynb
nbstripout-fast test_notebooks/*.ipynb
