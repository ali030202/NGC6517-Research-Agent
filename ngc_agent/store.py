"""Transactional local job ledger. One active training job per project."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

TERMINAL = {"succeeded", "failed", "cancelled", "timed_out", "interrupted"}


def now():
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.directory = self.root / ".agent"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "state.sqlite3"
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, kind TEXT NOT NULL,
                    state TEXT NOT NULL, config TEXT NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    worker_pid INTEGER, child_pid INTEGER, cancel_requested INTEGER NOT NULL DEFAULT 0,
                    source_run TEXT, proposal_id TEXT, result TEXT, error TEXT,
                    active_slot INTEGER NOT NULL DEFAULT 1
                );
                CREATE UNIQUE INDEX IF NOT EXISTS one_active_job
                    ON jobs(active_slot) WHERE state IN ('queued','running');
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL,
                    at TEXT NOT NULL, event TEXT NOT NULL, payload TEXT NOT NULL
                );
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def create(self, job_id, kind, config, source_run=None, proposal_id=None):
        with self.connect() as db:
            try:
                db.execute(
                    "INSERT INTO jobs(id,kind,state,config,created_at,updated_at,source_run,proposal_id) VALUES(?,?,'queued',?,?,?,?,?)",
                    (job_id, kind, json.dumps(config), now(), now(), source_run, proposal_id),
                )
                self._event(db, job_id, "queued", {"kind": kind})
            except sqlite3.IntegrityError as exc:
                raise ValueError("已有活动任务；请等待完成或取消后再提交。") from exc

    @staticmethod
    def _event(db, job_id, event, payload):
        db.execute("INSERT INTO events(job_id,at,event,payload) VALUES(?,?,?,?)",
                   (job_id, now(), event, json.dumps(payload, ensure_ascii=False)))

    def update(self, job_id, *, state=None, worker_pid=None, child_pid=None, result=None, error=None):
        values = {"updated_at": now()}
        if state is not None:
            values["state"] = state
        if worker_pid is not None:
            values["worker_pid"] = worker_pid
        if child_pid is not None:
            values["child_pid"] = child_pid
        if result is not None:
            values["result"] = json.dumps(result, ensure_ascii=False, allow_nan=False)
        if error is not None:
            values["error"] = error
        with self.connect() as db:
            row = db.execute("SELECT state FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not row:
                raise ValueError("Unknown job")
            if row["state"] in TERMINAL:
                if state is None:
                    return
                if state != row["state"]:
                    raise ValueError("Cannot change a completed job")
            db.execute(f"UPDATE jobs SET {', '.join(k+'=?' for k in values)} WHERE id=?", (*values.values(), job_id))
            if state is not None or error is not None:
                self._event(db, job_id, state or "error", {"error": error})

    def cancel(self, job_id):
        with self.connect() as db:
            row = db.execute("SELECT state FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not row:
                raise ValueError("Unknown job")
            if row["state"] in TERMINAL:
                return
            db.execute("UPDATE jobs SET cancel_requested=1,updated_at=? WHERE id=?", (now(), job_id))
            self._event(db, job_id, "cancel_requested", {})

    @staticmethod
    def decode(row):
        result = dict(row)
        for key in ("config", "result"):
            result[key] = json.loads(result[key]) if result[key] else None
        return result

    def get(self, job_id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            raise ValueError(f"Unknown job: {job_id}")
        return self.decode(row)

    def jobs(self, limit=100):
        with self.connect() as db:
            rows = db.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (min(max(int(limit), 1), 1000),)).fetchall()
        return [self.decode(row) for row in rows]

    def events(self, job_id):
        self.get(job_id)
        with self.connect() as db:
            rows = db.execute("SELECT * FROM events WHERE job_id=? ORDER BY seq", (job_id,)).fetchall()
        return [{**dict(row), "payload": json.loads(row["payload"])} for row in rows]

    def event(self, job_id, event, payload):
        with self.connect() as db:
            self._event(db, job_id, event, payload)

    def claim(self, job_id, pid):
        with self.connect() as db:
            changed = db.execute("UPDATE jobs SET state='running',worker_pid=?,updated_at=? WHERE id=? AND state='queued'", (pid, now(), job_id)).rowcount
            if changed != 1:
                raise ValueError("Worker can only claim a queued job once")
            self._event(db, job_id, "running", {"worker_pid": pid})
