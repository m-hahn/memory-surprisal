import os

from ud_languages import languages

import subprocess

for language in languages:
   subprocess.call(["./python27", "yAnalyzeNgrams.py", "1", language])

