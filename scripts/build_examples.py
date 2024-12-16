#!/usr/bin/env python3

import os
import subprocess
import yaml
import nbformat


def main():
    examples_dir = 'examples'
    output_dir = 'examples_executed'

    for filename in os.listdir(examples_dir):
        if filename.endswith('.ipynb'):
            input_path = os.path.join(examples_dir, filename)
            
            command = ['jupyter', 'nbconvert', '--execute', '--to', 'notebook', input_path, '--output-dir', output_dir]
            subprocess.run(command, check=True)

if __name__ == "__main__":
    main()
