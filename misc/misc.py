'''Misc functions on the go'''

import ast
import os

def get_imports(py_file):
    """Return a sorted set of imported modules from a .py file."""
    imports = set()

    try:
        with open(py_file, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=py_file)
    except (SyntaxError, UnicodeDecodeError):
        return imports  # skip broken or non-text files

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)

    return sorted(imports)

def print_tree_with_imports(start_path, prefix="", print_imports: bool=True):
    '''
    Docstring for print_tree_with_imports
    '''
    entries = sorted(os.listdir(start_path))
    entries = [e for e in entries if e != "__pycache__"]

    for index, name in enumerate(entries):
        path = os.path.join(start_path, name)
        is_last = index == len(entries) - 1
        connector = "└── " if is_last else "├── "

        if os.path.isdir(path):
            print(prefix + connector + name + "/")
            extension = "    " if is_last else "│   "
            print_tree_with_imports(path, prefix + extension)

        elif name.endswith(".py"):
            print(prefix + connector + name)

            if print_imports:
                imports = get_imports(path)
                for imp in imports:
                    print(prefix + ("    " if is_last else "│   ") + \
                           f"↳ import {imp}")

print_tree_with_imports('../src')
