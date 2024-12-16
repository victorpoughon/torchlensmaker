#!/usr/bin/env python3

import os
import re
import yaml
import nbformat


def get_h1(notebook_path):
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)

    first_cell_content = notebook.cells[0].source

    pattern = r"^#\s+(.*?)$"

    match = re.match(pattern, first_cell_content)
    if not match:
        raise RuntimeError("Cannot find h1 title in ", notebook_path)

    return match.group(1)


def main():
    examples_dir = "examples"
    output_dir = "examples/executed"

    all_examples = []

    for filename in os.listdir(examples_dir):
        if filename.endswith(".ipynb"):
            input_path = os.path.join(examples_dir, filename)
            h1 = get_h1(input_path)
            all_examples.append({h1: os.path.join(output_dir, filename)})

    print(
        yaml.dump(
            {"Examples": all_examples},
            width=1000,
            default_flow_style=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
