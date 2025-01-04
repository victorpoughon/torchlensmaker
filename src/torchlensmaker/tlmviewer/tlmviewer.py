from IPython.display import display, HTML
import string
import uuid
import os.path
import json


dir_path = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(dir_path, "script_template.js"), encoding="utf-8") as f:
    script_template = "<script type='module'>" + f.read() + "</script>"

div_template = "<div id='$div_id' style='width: 800px; height: 600px;'></div>"


def random_id():
    return f"tlmviewer-{uuid.uuid4().hex[:8]}"


def viewer(data):
    div_id = random_id()
    div = string.Template(div_template).substitute(div_id=div_id)
    script = string.Template(script_template).substitute(data=json.dumps(data, allow_nan=False), div_id=div_id)
    display(HTML(div + script))
