from __future__ import annotations

import re
import sys
from pathlib import Path


def get_version(path: Path) -> str:
    pattern = re.compile(r'^version\s*=\s*"([^"]+)"')
    try:
        text = path.read_text(encoding="utf8")
    except FileNotFoundError as exc:
        raise SystemExit(f"pyproject.toml not found at: {path}") from exc

    for line in text.splitlines():
        match = pattern.match(line.strip())
        if match:
            return match.group(1)

    raise SystemExit(f"Could not find version in {path}")


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        path = Path("pyproject.toml")
    else:
        path = Path(argv[0])

    version = get_version(path)
    print(version)


if __name__ == "__main__":
    main()
