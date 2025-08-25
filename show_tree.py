from treelib import Tree
import os

# Folders and files to ignore
IGNORE_DIRS = {
    "venv", "env", "__pycache__", ".git", ".idea", ".vscode",
    "dist", "build", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", "htmlcov"
}
IGNORE_FILES = {".DS_Store", ".coverage"}
IGNORE_EXTENSIONS = {".pyc", ".pyo", ".log", ".egg-info"}

def build_tree(path, tree, parent=None):
    for item in sorted(os.listdir(path)):
        # Skip ignored dirs/files
        if item in IGNORE_DIRS:
            continue
        if item in IGNORE_FILES:
            continue
        if any(item.endswith(ext) for ext in IGNORE_EXTENSIONS):
            continue

        full_path = os.path.join(path, item)
        if os.path.isdir(full_path):
            tree.create_node(item, full_path, parent=parent)
            build_tree(full_path, tree, parent=full_path)
        else:
            tree.create_node(item, full_path, parent=parent)

def show_project_structure(base_path="."):
    tree = Tree()
    root_name = os.path.basename(os.path.abspath(base_path))
    tree.create_node(root_name, base_path)  # root
    build_tree(base_path, tree, parent=base_path)
    tree.show()

if __name__ == "__main__":
    show_project_structure(".")