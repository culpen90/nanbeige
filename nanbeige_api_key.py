"""Persistent bearer key for the local Nanbeige API."""

import os
from pathlib import Path
import re
import secrets
import stat
from tempfile import NamedTemporaryFile


def _read_key(path: Path) -> str:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as file:
        mode = os.fstat(file.fileno()).st_mode
        if not stat.S_ISREG(mode) or mode & 0o077:
            raise ValueError(f"Expected an owner-only regular API key file at {path}.")
        stored = file.read(128)
    if not re.fullmatch(rb"sk-[0-9a-f]{64}\n?", stored):
        raise ValueError(f"Invalid Nanbeige API key file at {path}.")
    return stored.rstrip(b"\n").decode("ascii")


def get_api_key() -> str:
    config = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    path = config / "nanbeige" / "api-key"
    try:
        return _read_key(path)
    except FileNotFoundError:
        pass

    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    key = "sk-" + secrets.token_hex(32)
    with NamedTemporaryFile(dir=path.parent, mode="w", encoding="ascii") as temporary:
        temporary.write(key + "\n")
        temporary.flush()
        os.fsync(temporary.fileno())
        # Publish a complete, private file only if another process hasn't won the race.
        try:
            os.link(temporary.name, path)
        except FileExistsError:
            return _read_key(path)
    return key
