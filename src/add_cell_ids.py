#!/usr/bin/env python3
import sys, uuid
from pathlib import Path
import nbformat as nbf

def add_ids(path: Path) -> bool:
    nb = nbf.read(path, as_version=4)
    changed = False
    for c in nb.cells:
        if not getattr(c, "id", None):
            # nbformat requires a short, unique string id
            c.id = uuid.uuid4().hex[:12]
            changed = True
    if changed:
        nbf.write(nb, path)
    return changed

def main(paths):
    for p in map(Path, paths):
        if p.suffix == ".ipynb" and p.exists():
            try:
                if add_ids(p):
                    print(f"[add_cell_ids] fixed: {p}")
            except Exception as e:
                # Print the error and exit non-zero so you can see which file failed
                print(f"[add_cell_ids] ERROR processing {p}: {e}", file=sys.stderr)
                return 1
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
