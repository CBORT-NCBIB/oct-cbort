import os

if os.name == 'nt':
    os.system(
        "open docs/_build/html/index.html")
elif os.name == 'posix':
    os.system(
        "open docs/_build/html/index.html")

