"""Stage inspectable model proposals and preserve content-addressed backups."""
import ast
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import uuid


def digest(content):
    return hashlib.sha256(content).hexdigest()


def atomic_text(path, text):
    atomic_bytes(path, text.encode("utf-8"))


def atomic_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_model(code):
    tree = ast.parse(code, filename="model_proto.py")
    encoder = next((node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NGC6517Encoder"), None)
    logits = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "prototypical_logits"), None)
    if encoder is None or logits is None:
        raise ValueError("模型必须导出 NGC6517Encoder 和 prototypical_logits")
    init = next((node for node in encoder.body if isinstance(node, ast.FunctionDef) and node.name == "__init__"), None)
    forward = next((node for node in encoder.body if isinstance(node, ast.FunctionDef) and node.name == "forward"), None)
    if init is None or forward is None:
        raise ValueError("NGC6517Encoder requires explicit __init__ and forward methods")
    names = {arg.arg for arg in init.args.args + init.args.kwonlyargs}
    if init.args.kwarg is None and not {"embed_dim", "dropout", "use_prior"} <= names:
        raise ValueError("Encoder constructor must accept embed_dim, dropout, use_prior")
    names = {arg.arg for arg in logits.args.args + logits.args.kwonlyargs}
    if logits.args.kwarg is None and not {"temperature", "psr_k", "rfi_k"} <= names:
        raise ValueError("prototypical_logits must accept temperature, psr_k, rfi_k")
    return {"syntax": "valid", "interface": "static checks passed", "limitation": "静态检查不执行代码，也不证明张量形状、数值正确性或代码安全性。"}


class CodeManager:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.target = self.root / "model_proto.py"
        self.directory = self.root / ".agent" / "proposals"

    def path(self, proposal_id):
        if not re.fullmatch(r"[a-f0-9]{32}", proposal_id):
            raise ValueError("Invalid proposal ID")
        return self.directory / proposal_id

    def stage(self, code, rationale=""):
        validation = validate_model(code)
        original = self.target.read_text(encoding="utf-8-sig")
        proposal_id = uuid.uuid4().hex
        folder = self.path(proposal_id)
        folder.mkdir(parents=True)
        atomic_text(folder / "model_proto.py", code)
        patch = "".join(difflib.unified_diff(original.splitlines(True), code.splitlines(True), fromfile="current/model_proto.py", tofile="proposal/model_proto.py"))
        atomic_text(folder / "changes.diff", patch)
        manifest = {"id": proposal_id, "base_sha256": digest(self.target.read_bytes()),
                    "code_sha256": digest((folder / "model_proto.py").read_bytes()),
                    "rationale": rationale, "validation": validation, "state": "staged"}
        atomic_text(folder / "proposal.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        return {**manifest, "diff_path": str(folder / "changes.diff"), "code_path": str(folder / "model_proto.py")}

    def read(self, proposal_id):
        folder = self.path(proposal_id)
        manifest = json.loads((folder / "proposal.json").read_text(encoding="utf-8"))
        code_path = folder / "model_proto.py"
        if digest(code_path.read_bytes()) != manifest["code_sha256"]:
            raise ValueError("Proposal changed after staging; stage it again")
        validate_model(code_path.read_text(encoding="utf-8"))
        return manifest, code_path

    def backup(self):
        current = self.target.read_bytes()
        sha = digest(current)
        backup = self.root / ".agent" / "backups" / f"{sha}.py"
        # Preserve the exact bytes so the backup's filename is its real digest.
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            atomic_bytes(backup, current)
        return sha

    def apply(self, proposal_id):
        manifest, path = self.read(proposal_id)
        if digest(self.target.read_bytes()) != manifest["base_sha256"]:
            raise ValueError("当前模型已改变；请重新生成差异后再应用该提案。")
        backup_sha = self.backup()
        atomic_text(self.target, path.read_text(encoding="utf-8"))
        manifest.update(state="applied", backup_sha256=backup_sha)
        atomic_text(path.parent / "proposal.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        return manifest

    def rollback(self, sha):
        if not re.fullmatch(r"[a-f0-9]{64}", sha):
            raise ValueError("Invalid backup SHA256")
        backup = self.root / ".agent" / "backups" / f"{sha}.py"
        content = backup.read_bytes()
        if digest(content) != sha:
            raise ValueError("Backup checksum mismatch")
        validate_model(content.decode("utf-8-sig"))
        previous = self.backup()
        atomic_bytes(self.target, content)
        return {"state": "restored", "restored_sha256": digest(self.target.read_bytes()), "previous_backup": previous}
