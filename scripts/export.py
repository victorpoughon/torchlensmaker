#!/usr/bin/env python3

import os.path
from pathlib import Path
import argparse
import nbformat

from nbconvert import nbconvertapp

from nbconvert import MarkdownExporter
from nbconvert.preprocessors import (
    ExecutePreprocessor,
    RegexRemovePreprocessor,
    CoalesceStreamsPreprocessor,
)
from nbconvert.writers import FilesWriter



def print_md_list(folder: Path):
    for notebook in (Path("docs") / folder).glob("*.md"):
        path = Path(*notebook.parts[1:])
        print(f"* [{path}]({path})")


def main():
    parser = argparse.ArgumentParser(
        prog="export.py",
        description="export notebooks to markdown for inclusion in the docs",
    )

    parser.add_argument("filepath")
    parser.add_argument(
        "-s",
        "--skip",
        action="store_true",
        help="Skip any notebook that already exists at the destination",
    )
    parser.add_argument(
        "-p",
        "--print_md_list",
        action="store_true",
        help="Print list at destination in md format",
    )

    args = parser.parse_args()

    dir_path = Path(__file__).resolve().parent.parts
    fullpath = Path(args.filepath).resolve().parts

    if Path(args.filepath).is_file():
        output_folder_relative = Path(fullpath[-2])
        output_folder = Path(*dir_path[:-1]) / "docs" / output_folder_relative
        print(f"Exporting notebook {args.filepath} to {output_folder}")
        export_notebook(Path(args.filepath), output_folder, args.skip)
    elif Path(args.filepath).is_dir():
        output_folder_relative = Path(fullpath[-1])
        output_folder = Path(*dir_path[:-1]) / "docs" / output_folder_relative
        print(f"Exporting all notebooks in {args.filepath} to {output_folder}")
        export_all(Path(args.filepath), output_folder, args.skip)
    
    # Print markdown list if requested
    if args.print_md_list:
        print("Markdown format list:")
        print_md_list(output_folder_relative)


def export_all(filename: Path, output_folder: Path, skip: bool) -> None:
    """Export all notebooks from folder"""

    # Ensure the output folder exists
    output_folder.mkdir(parents=True, exist_ok=True)

    # Iterate over all .ipynb files in the directory
    for notebook in filename.glob("*.ipynb"):
        export_notebook(notebook, output_folder, skip)


def export_notebook(filename: Path, output_folder: Path, skip: bool) -> None:
    "Export a single notebook"
    root = filename.stem

    if skip and os.path.exists(output_folder / (root + ".md")):
        print(f".. skiping (destination exists) {filename}")
        return
    else:
        print(f"> {filename}")

    with open(filename) as f:
        nb = nbformat.read(f, as_version=4)

    resources = nbconvertapp.NbConvertApp().init_single_notebook_resources(filename)

    processors = [
        # Execute the notebook
        ExecutePreprocessor(timeout=600, kernel_name="python3"),
        # Merge consecutive sequences of stream output into single stream to
        # prevent extra newlines inserted at flush calls
        CoalesceStreamsPreprocessor(),
        # Remove cells containing only whitespace or empty
        RegexRemovePreprocessor(patterns=[r"\s*\Z"]),
    ]

    for ep in processors:
        ep.preprocess(nb, resources)

    # Convert the notebook to Markdown
    md_exporter = MarkdownExporter()
    (body, resources) = md_exporter.from_notebook_node(nb, resources)

    # Write the output to a specific directory
    writer = FilesWriter()
    writer.build_directory = str(output_folder)

    writer.write(body, resources, notebook_name=root)


if __name__ == "__main__":
    main()
