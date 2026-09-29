"""Thin fsspec wrapper over the lake root (local directory, R2, B2, ...).

Keys are ``/``-separated paths relative to the lake root. Writes are create-only by
default so bronze can never be silently overwritten.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import fsspec


class Lake:
    def __init__(self, uri: str, **storage_options: Any) -> None:
        self.uri = uri
        self.fs, root = fsspec.core.url_to_fs(uri, **storage_options)
        self.root = root.rstrip("/")
        self._local = "file" in _protocols(self.fs)
        if self._local:
            self.fs.makedirs(self.root, exist_ok=True)

    def path(self, key: str) -> str:
        key = key.lstrip("/")
        if not key or ".." in key.split("/"):
            raise ValueError(f"invalid lake key: {key!r}")
        return f"{self.root}/{key}"

    def exists(self, key: str) -> bool:
        return bool(self.fs.exists(self.path(key)))

    def put_bytes(self, key: str, data: bytes, *, overwrite: bool = False) -> str:
        """Write ``data`` at ``key``. Raises FileExistsError unless ``overwrite``."""
        target = self.path(key)
        if not overwrite and self.fs.exists(target):
            raise FileExistsError(f"lake object already exists: {key}")
        parent = target.rsplit("/", 1)[0]
        if self._local:
            self.fs.makedirs(parent, exist_ok=True)
            tmp = f"{parent}/.tmp-{uuid.uuid4().hex}"
            with open(tmp, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            if overwrite:
                os.replace(tmp, target)
            else:
                # os.link fails if target exists: atomic create-only publish.
                try:
                    os.link(tmp, target)
                finally:
                    os.unlink(tmp)
        else:
            self.fs.pipe_file(target, data)
        return key

    def get_bytes(self, key: str) -> bytes:
        data = self.fs.cat_file(self.path(key))
        assert isinstance(data, bytes)
        return data

    def list(self, prefix: str = "") -> list[str]:
        """All object keys under ``prefix`` (recursive), sorted, relative to the root."""
        base = self.path(prefix) if prefix else self.root
        if not self.fs.exists(base):
            return []
        found = self.fs.find(base)
        root = self.root + "/"
        keys = [str(p)[len(root) :] if str(p).startswith(root) else str(p) for p in found]
        return sorted(k for k in keys if not k.rsplit("/", 1)[-1].startswith(".tmp-"))


def _protocols(fs: Any) -> tuple[str, ...]:
    proto = fs.protocol
    return (proto,) if isinstance(proto, str) else tuple(proto)
