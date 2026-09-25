"""Workspace checks and API documentation derived from actual Python sources."""

from pathlib import Path
import ast
import json
import re

from .storage import file_digest

REQUIRED_DOCS = ("START_HERE.md", "AGENTS.md", "Documentation_template.md", "docs/IMPLEMENTATION_GUIDE.md",
                 "docs/DATA_RESEARCH_REPORT.md", "docs/FUNCTION_REFERENCE.md", "docs/IMPLEMENTATION_STATUS.md",
                 "docs/BASELINE_IMPLEMENTATION.md", "docs/RUNBOOK.md", "docs/approach.md")


def write_function_reference(workspace):
    """Document every source function/method from AST signatures and full docstrings."""
    workspace = Path(workspace).resolve()
    source_root = workspace / "code/business_entity_resolution/src"
    lines = ["# Implemented function reference", "",
             "Generated from the actual Python source. These functions exist; optional future designs remain in the implementation guide.",
             "", "The runbook describes command order. Functions named with an underscore are internal helpers. Legacy aligned helpers remain available, but the command line uses strict disk-backed scoring/validation.", ""]
    for path in sorted(source_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        lines.extend(["## " + str(path.relative_to(source_root)).replace("\\", "/"), "", ast.get_docstring(tree) or "", ""])
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                lines.extend(["### `" + node.name + "`", "", "```python", node.name + "(" + ast.unparse(node.args) + ")", "```", "", ast.get_docstring(node) or "Missing docstring", ""])
            elif isinstance(node, ast.ClassDef):
                lines.extend(["### `" + node.name + "`", "", ast.get_docstring(node) or "", ""])
                for method in node.body:
                    if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        lines.extend(["#### `" + node.name + "." + method.name + "`", "", "```python", method.name + "(" + ast.unparse(method.args) + ")", "```", "", ast.get_docstring(method) or "Missing docstring", ""])
    target = workspace / "docs/FUNCTION_REFERENCE.md"
    target.write_text("\n".join(lines), encoding="utf-8")
    return {"path": str(target), "source_files": len(list(source_root.rglob("*.py")))}


def check_workspace(workspace, input_root=None):
    """Check required docs, local links, source syntax/docstrings and original hashes.

    Read only explicit project files and paths from the audited input manifest.
    No model quality, resource quota or full-data runtime guarantee is implied.
    """
    workspace = Path(workspace).resolve(strict=True)
    input_root = Path(input_root).resolve(strict=True) if input_root else workspace
    errors, checks = [], {}
    package = workspace / "code/business_entity_resolution"
    required = REQUIRED_DOCS + tuple("code/business_entity_resolution/" + name for name in
                                    ("README.md", "requirements.txt", "requirements-dev.txt", "configs/baseline.json", "src/run.py", "LICENSE", "THIRD_PARTY_NOTICES.md"))
    for name in required:
        if not (workspace / name).is_file():
            errors.append("Missing required file: " + name)
    markdown = [workspace / "START_HERE.md", workspace / "AGENTS.md", workspace / "Documentation_template.md",
                package / "README.md", package / "THIRD_PARTY_NOTICES.md"]
    markdown += sorted((workspace / "docs").glob("*.md"))
    for path in markdown:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)]+)\)", text):
            target = target.strip("<>")
            if "://" in target or target.startswith("#"):
                continue
            local = re.sub(r":\d+$", "", target.split("#", 1)[0])
            if local and not (path.parent / local).exists():
                errors.append(f"Broken link in {path.name}: {target}")
    checks["markdown_files"] = len(markdown)
    source_paths = sorted((package / "src").rglob("*.py"))
    function_count = 0
    for path in source_paths:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    function_count += isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    if not ast.get_docstring(node):
                        errors.append(f"Undocumented symbol: {path.name}:{node.name}")
        except SyntaxError as exc:
            errors.append(f"Syntax error in {path.name}: {exc}")
    checks["source_files"] = len(source_paths)
    checks["documented_functions_and_methods"] = function_count
    manifest_path = workspace / "docs/research/input_manifest.json"
    if not manifest_path.is_file():
        errors.append("Missing original input manifest")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        verified = 0
        for name, expected in manifest.items():
            path = (input_root / name).resolve()
            if not path.is_relative_to(input_root):
                errors.append("Input manifest path escapes workspace")
                continue
            if not path.is_file() or path.stat().st_size != expected["bytes"] or file_digest(path) != expected["sha256"]:
                errors.append("Original input changed/missing: " + name)
            else:
                verified += 1
        checks["original_inputs_hash_verified"] = verified
    from .config import load_config
    try:
        load_config(package / "configs/baseline.json")
        checks["baseline_config_valid"] = True
    except (ValueError, OSError) as exc:
        errors.append("Baseline configuration: " + str(exc))
    return {"passed": not errors, "checks": checks, "errors": errors,
            "scope": "Code/docs/input integrity. Model performance and full-scale runtime require separate measurements."}
