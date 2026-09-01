from __future__ import annotations
import hashlib
import importlib.util
from pathlib import Path

BASE = Path(__file__).with_name('build_intelligence_console_v4_model_lab.py')
spec = importlib.util.spec_from_file_location('assetgraph_v4_base', BASE)
mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(mod)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

# v4 imports the v3 wrapper as `mod`; the SHA helper lives lower in the builder stack.
# Inject a local, deterministic helper so inference code is independent from wrapper internals.
mod.mod.sha256_file = sha256_file
mod.main()
