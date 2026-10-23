"""Compatibility facade for inspectable model proposals and versioned rollback."""
from pathlib import Path
from ngc_agent.code import CodeManager
from ngc_agent.config import ROOT


def _manager(file_name):
    if file_name != "model_proto.py":
        raise ValueError("Only model_proto.py can be changed through model tools")
    return CodeManager(ROOT)


def modify_model_code(file_name="model_proto.py", modify_prompt=""):
    manager = _manager(file_name)
    sha = manager.backup()
    return {"original_code": manager.target.read_text(encoding="utf-8-sig"),
            "backup_file": str(ROOT / ".agent" / "backups" / f"{sha}.py"),
            "backup_sha256": sha, "modify_prompt": modify_prompt}


def write_model_code(file_name, new_full_code):
    try:
        manager = _manager(file_name)
        proposal = manager.stage(new_full_code, "Legacy research workflow")
        applied = manager.apply(proposal["id"])
        return f"Applied proposal {proposal['id']}; backup={applied['backup_sha256']}"
    except (ValueError, SyntaxError, OSError) as exc:
        return f"error: {exc}"


def rollback_model_code(file_name="model_proto.py", backup_sha256=None):
    manager = _manager(file_name)
    if not backup_sha256:
        backups = sorted((ROOT / ".agent" / "backups").glob("*.py"), key=lambda path: path.stat().st_mtime)
        if not backups:
            return "error: no backup exists"
        backup_sha256 = backups[-1].stem
    return manager.rollback(backup_sha256)
