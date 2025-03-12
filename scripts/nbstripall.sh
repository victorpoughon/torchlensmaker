#!/usr/bin/env bash

set -euo pipefail

nbstripout-fast examples/*.ipynb
nbstripout-fast test_notebooks/*.ipynb
nbstripout-fast docs/src/*.ipynb
