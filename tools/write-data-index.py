#!/usr/bin/env python3
"""Writes the index of game data files the browser client asks for.

The client mounts Data/ and downloads each file as it is read, but it has to
be told what is there first: a relative path, a tab and the size in bytes, a
line (see src/platform/web_data.cpp). tools/serve-wasm.py answers that from
the directory on every request; a hosted server serves a file instead, and
this writes it.

    tools/write-data-index.py /srv/wowee/Data

Run it again whenever the data changes.
"""
import os
import sys


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    root = os.path.abspath(sys.argv[1])
    if not os.path.isdir(root):
        sys.exit(f"{root} is not a directory")

    lines = []
    for dirpath, _, filenames in os.walk(root, followlinks=True):
        rel = os.path.relpath(dirpath, root)
        for name in filenames:
            if name.startswith(".wowee-index"):
                continue
            try:
                size = os.stat(os.path.join(dirpath, name)).st_size
            except OSError:
                continue
            lines.append(f"{name if rel == '.' else f'{rel}/{name}'}\t{size}")
    lines.sort()

    # Written beside the data it describes, and replaced in one step so a
    # client reading it never sees half a list.
    index = os.path.join(root, ".wowee-index")
    with open(index + ".new", "w") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(index + ".new", index)
    print(f"{index}: {len(lines)} files")


if __name__ == "__main__":
    main()
