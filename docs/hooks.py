# from: https://github.com/squidfunk/mkdocs-material/discussions/3458

import os
import shutil

def copy_get(config, **kwargs):
    site_dir = config['site_dir']
    shutil.copy('docs/tlmviewer.js', os.path.join(site_dir, 'tlmviewer.js'))