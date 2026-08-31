from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CacheKey:
    namespace: str
    inputs: tuple[tuple[str, str], ...]

    def digest(self) -> str:
        encoded = json.dumps(
            {"namespace": self.namespace, "inputs": sorted(self.inputs)},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(encoded.encode()).hexdigest()


class ContentAddressedCache:
    def __init__(self, root: Path) -> None:
        self.root = root

    def path(self, key: CacheKey) -> Path:
        return self.root / key.namespace / key.digest()

    def write_bytes(self, key: CacheKey, value: bytes) -> Path:
        target = self.path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(value).hexdigest()
        temporary = Path(f"{target}.data.tmp")
        sidecar = Path(f"{target}.sha256")
        sidecar_temporary = Path(f"{sidecar}.tmp")
        temporary.write_bytes(value)
        sidecar_temporary.write_text(digest + "\n", encoding="utf-8")
        temporary.replace(target)
        sidecar_temporary.replace(sidecar)
        return target

    def read_bytes(self, key: CacheKey) -> bytes | None:
        target = self.path(key)
        sidecar = Path(f"{target}.sha256")
        if not target.exists() or not sidecar.exists():
            return None
        value = target.read_bytes()
        expected = sidecar.read_text(encoding="utf-8").strip()
        if hashlib.sha256(value).hexdigest() != expected:
            raise ValueError(f"cache integrity failure: {target}")
        return value
