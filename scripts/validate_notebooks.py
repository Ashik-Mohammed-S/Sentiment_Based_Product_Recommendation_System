"""Validate notebook syntax and dependency imports without executing datasets or training."""
import ast
import importlib
from pathlib import Path
import nbformat
from IPython.core.inputtransformer2 import TransformerManager

root = Path(__file__).resolve().parents[1]
transformer = TransformerManager()
modules = set()
from_imports = set()
count = 0
for path in root.glob("*.ipynb"):
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    for index, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        code = transformer.transform_cell(cell.source)
        compile(code, f"{path.name}:cell{index}", "exec")
        for node in ast.walk(ast.parse(code)):
            if isinstance(node, ast.Import):
                modules.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module.split(".")[0])
                from_imports.update((node.module, alias.name) for alias in node.names if alias.name != "*")
        count += 1
for module in sorted(modules):
    importlib.import_module(module)
for module, name in sorted(from_imports):
    imported = importlib.import_module(module)
    if not hasattr(imported, name):
        importlib.import_module(f"{module}.{name}")
print(f"Validated {count} code cells; imported {len(modules)} modules.")
