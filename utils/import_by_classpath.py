import importlib

def import_by_modulepath(module_path):
    module = importlib.import_from_string(module_path)
    return module