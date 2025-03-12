#!/usr/bin/env python3

import os
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
    for notebook in (Path("docs") / Path("src") / folder).glob("*.md"):
        path = Path(*notebook.parts[2:])
        print(f"* [{path}]({path})")


def main():
    parser = argparse.ArgumentParser(
        prog="export.py",
        description="export notebooks to markdown for inclusion in the docs",
    )

    parser.add_argument("filepaths", nargs='+')
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

    for filepath in args.filepaths:

        # dir_path = Path(__file__).resolve().parent.parts
        fullpath = Path(filepath).resolve()

        if Path(filepath).is_file():
            output_folder = fullpath.parent
            print(f"Exporting notebook {filepath} to {output_folder}")
            export_notebook(Path(filepath), output_folder, args.skip)
        elif Path(filepath).is_dir():
            output_folder = fullpath
            print(f"Exporting all notebooks in {filepath} to {output_folder}")
            export_all(Path(filepath), output_folder, args.skip)
        else:
            raise RuntimeError(f"{filepath} not found")
        
        # Print markdown list if requested
        if args.print_md_list:
            print("Markdown format list:")
            print_md_list(output_folder)


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
    
    # Set env var to request vue component format
    os.environ["TLMVIEWER_TARGET_DIRECTORY"] = str(output_folder)
    os.environ["TLMVIEWER_TARGET_NAME"] = root
    os.environ["TLMVIEWER_TARGET_FORMAT"] = "vue"

    resources = nbconvertapp.NbConvertApp().init_single_notebook_resources(filename)

    # fmt: off
    processors = [
        # Execute the notebook
        ExecutePreprocessor(timeout=600, kernel_name="python3"),

        # Merge consecutive sequences of stream output into single stream to
        # prevent extra newlines inserted at flush calls
        CoalesceStreamsPreprocessor(),
        
        # Remove cells containing only whitespace or empty
        RegexRemovePreprocessor(patterns=[r"\s*\Z"]),
    ]
    # fmt: on

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
