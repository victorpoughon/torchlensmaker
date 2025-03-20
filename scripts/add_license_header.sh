#!/usr/bin/env bash

set -euo pipefail

directory="$1"
header_file=$(realpath "./scripts/license_header.py")

cd "$directory"

find . -type f -name "*.py" | while read -r file; do
    cat "$header_file" "$file" > temp && mv temp "$file"
done

