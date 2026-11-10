"""The implementation uses Lucid's public API and the standard library, nothing else.

Fails on a third-party import, on any ``lucid._…`` import or attribute access,
and on the model zoo (``lucid.models``).
"""

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "remind_lucid"
FORBIDDEN_TOP = {
    "torch", "numpy", "np", "scipy", "PIL", "cv2", "einops", "safetensors",
    "transformers", "diffusers", "tokenizers", "yaml", "mlx", "ftfy", "regex",
    "imageio", "omegaconf", "peft",
}  # fmt: skip
LUCID_ROOTS = {"lucid", "nn", "F"}  # the names the package binds Lucid modules to


def _imports(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module
            for alias in node.names:
                yield f"{node.module}.{alias.name}"


def _attribute_chain(node: ast.Attribute) -> list[str] | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return parts[::-1]
    return None


def _private_lucid_attributes(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            chain = _attribute_chain(node)
            if not chain or chain[0] not in LUCID_ROOTS:
                continue
            if any(p.startswith("_") and not p.startswith("__") for p in chain[1:]):
                yield ".".join(chain)
            if chain[:2] == ["lucid", "models"]:
                yield ".".join(chain)


def test_no_third_party_or_private_lucid():
    offenders = []
    for path in PACKAGE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for name in _imports(tree):
            parts = name.split(".")
            if parts[0] in FORBIDDEN_TOP:
                offenders.append((path.name, name))
            if parts[0] == "lucid" and any(p.startswith("_") for p in parts[1:]):
                offenders.append((path.name, name))
            if parts[:2] == ["lucid", "models"]:
                offenders.append((path.name, name))
        offenders += [(path.name, a) for a in _private_lucid_attributes(tree)]
    assert not offenders, offenders
