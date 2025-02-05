#!/usr/bin/env python3

import os.path
import argparse
import nbformat

from nbconvert import MarkdownExporter
from nbconvert.preprocessors import ExecutePreprocessor, RegexRemovePreprocessor
from nbconvert.writers import FilesWriter


def main():
    parser = argparse.ArgumentParser(prog='export.py', description='"')

    parser.add_argument('filename')
    args = parser.parse_args()

    print("Exporting", args.filename)
    export_notebook(args.filename)


def export_notebook(filename: str) -> None:
    head, tail = os.path.split(filename)
    root, ext = os.path.splitext(tail)

    with open(filename) as f:
        nb = nbformat.read(f, as_version=4)

    processors = [
        # Execute the notebook
        ExecutePreprocessor(timeout=600, kernel_name='python3'),

        # Remove cells containing only whitespace or empty
        RegexRemovePreprocessor(patterns=[r"\s*\Z"]),
    ]

    for ep in processors:
        ep.preprocess(nb, resources={})

    # Convert the notebook to Markdown
    html_exporter = MarkdownExporter()
    (body, resources) = html_exporter.from_notebook_node(nb)

    # Write the output to a specific directory
    writer = FilesWriter()
    writer.build_directory = "docs/examples/"
    writer.write(body, resources, notebook_name=root)


if __name__ == "__main__":
    main()
