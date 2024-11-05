#!/usr/bin/env bash

set -euo pipefail

jupyter nbconvert --clear-output --inplace examples/*.ipynb test_notebooks/*.ipynb