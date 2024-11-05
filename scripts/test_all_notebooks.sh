#!/usr/bin/env bash

set -euo pipefail

pytest --nbmake **/*ipynb
