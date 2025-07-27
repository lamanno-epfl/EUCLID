#!/usr/bin/env python3
import os, ast
try:
    # use the new stdlib API first
    from importlib.metadata import version as get_version
except ImportError:
    # fallback for older Python
    from pkg_resources import get_distribution as get_version

SRC_DIR = "src"

def find_imports(base_dir):
    pkgs = set()
    for root, _, files in os.walk(base_dir):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(root, fn)
            try:
                src = open(path, encoding="utf8").read()
                tree = ast.parse(src, path)
            except Exception:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for n in node.names:
                        pkgs.add(n.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        pkgs.add(node.module.split(".")[0])
    return pkgs

def pin_versions(pkgs):
    pinned = []
    for pkg in sorted(pkgs):
        # skip names that aren’t real packages
        if pkg.startswith("_") or pkg in ("__future__",):
            continue
        try:
            ver = get_version(pkg)
            pinned.append(f"{pkg}=={ver}")
        except Exception:
            # any failure → skip
            continue
    return pinned

if __name__ == "__main__":
    imports = find_imports(SRC_DIR)
    pinned = pin_versions(imports)
    with open("imports.txt", "w") as f:
        f.write("\n".join(pinned) + "\n")
    print(f"Wrote {len(pinned)} dependencies to imports.txt")

