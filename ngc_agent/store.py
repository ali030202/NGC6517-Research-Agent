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
                CREATE TABLE IF NOT EXISTS workflows (
                    id TEXT PRIMARY KEY, state TEXT NOT NULL, phase TEXT NOT NULL,
                    request TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    worker_pid INTEGER, active_job_id TEXT, cancel_requested INTEGER NOT NULL DEFAULT 0,
                    error TEXT, report_path TEXT
                );
                CREATE UNIQUE INDEX IF NOT EXISTS one_active_workflow
                    ON workflows((1)) WHERE state IN ('queued','running');
            """)
            columns = {row["name"] for row in db.execute("PRAGMA table_info(jobs)")}
            for name in ("request_key", "workflow_id"):
                if name not in columns:
                    try:
                        db.execute(f"ALTER TABLE jobs ADD COLUMN {name} TEXT")
                    except sqlite3.OperationalError:
                        if name not in {row["name"] for row in db.execute("PRAGMA table_info(jobs)")}:
                            raise
            db.execute("CREATE UNIQUE INDEX IF NOT EXISTS unique_job_request ON jobs(request_key)")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def create(self, job_id, kind, config, source_run=None, proposal_id=None,
               request_key=None, workflow_id=None):
        with self.connect() as db:
            try:
                db.execute(
                    "INSERT INTO jobs(id,kind,state,config,created_at,updated_at,source_run,proposal_id,request_key,workflow_id) VALUES(?,?,'queued',?,?,?,?,?,?,?)",
                    (job_id, kind, json.dumps(config), now(), now(), source_run, proposal_id, request_key, workflow_id),
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

    def record_job_launch(self, job_id, expected_pid, pid=None, error=None):
        with self.connect() as db:
            if error is None:
                db.execute("UPDATE jobs SET worker_pid=?,updated_at=? WHERE id=? AND state='queued' AND worker_pid IS ?",
                           (pid, now(), job_id, expected_pid))
            else:
                changed = db.execute("UPDATE jobs SET state='failed',error=?,updated_at=? WHERE id=? AND state='queued' AND worker_pid IS ?",
                                     (error, now(), job_id, expected_pid)).rowcount
                if changed:
                    self._event(db, job_id, "failed", {"error": error})

    def find_request(self, request_key):
        with self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE request_key=?", (request_key,)).fetchone()
        return self.decode(row) if row else None

    def workflow_jobs(self, workflow_id):
        with self.connect() as db:
            rows = db.execute("SELECT * FROM jobs WHERE workflow_id=? ORDER BY created_at", (workflow_id,)).fetchall()
        return [self.decode(row) for row in rows]

    def create_workflow(self, workflow_id, request):
        with self.connect() as db:
            try:
                db.execute("INSERT INTO workflows(id,state,phase,request,created_at,updated_at) VALUES(?,'queued','created',?,?,?)",
                           (workflow_id, json.dumps(request, ensure_ascii=False), now(), now()))
                self._event(db, workflow_id, "workflow_created", {})
            except sqlite3.IntegrityError as exc:
                raise ValueError("已有活动工作流；请等待完成、暂停或取消后再启动。") from exc

    def workflow(self, workflow_id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM workflows WHERE id=?", (workflow_id,)).fetchone()
        if row is None:
            raise ValueError(f"Unknown workflow: {workflow_id}")
        result = dict(row)
        result["request"] = json.loads(result["request"])
        return result

    def workflows(self, limit=50):
        with self.connect() as db:
            rows = db.execute("SELECT id FROM workflows ORDER BY created_at DESC LIMIT ?", (min(max(int(limit), 1), 1000),)).fetchall()
        return [self.workflow(row["id"]) for row in rows]

    def update_workflow(self, workflow_id, **values):
        allowed = {"state", "phase", "worker_pid", "active_job_id", "error", "report_path"}
        if not values or set(values) - allowed:
            raise ValueError("Invalid workflow update")
        values["updated_at"] = now()
        with self.connect() as db:
            changed = db.execute(f"UPDATE workflows SET {', '.join(key+'=?' for key in values)} WHERE id=?",
                                 (*values.values(), workflow_id)).rowcount
            if changed != 1:
                raise ValueError("Unknown workflow")

    def claim_workflow(self, workflow_id, pid):
        with self.connect() as db:
            changed = db.execute("UPDATE workflows SET state='running',worker_pid=?,updated_at=? WHERE id=? AND state='queued'",
                                 (pid, now(), workflow_id)).rowcount
            if changed:
                self._event(db, workflow_id, "workflow_running", {"worker_pid": pid})
        return changed == 1

    def record_workflow_launch(self, workflow_id, expected_pid, pid=None, error=None):
        with self.connect() as db:
            if error is None:
                db.execute("UPDATE workflows SET worker_pid=?,updated_at=? WHERE id=? AND state='queued' AND worker_pid IS ?",
                           (pid, now(), workflow_id, expected_pid))
            else:
                db.execute("UPDATE workflows SET state='failed',error=?,updated_at=? WHERE id=? AND state='queued' AND worker_pid IS ?",
                           (error, now(), workflow_id, expected_pid))

    def requeue_workflow(self, workflow_id, expected_updated_at, expected_pid):
        with self.connect() as db:
            changed = db.execute("""UPDATE workflows SET state='queued',worker_pid=NULL,error=NULL,updated_at=?
                WHERE id=? AND updated_at=? AND worker_pid IS ? AND cancel_requested=0
                AND state NOT IN ('succeeded','cancelled','timed_out')""",
                                 (now(), workflow_id, expected_updated_at, expected_pid)).rowcount
        return changed == 1

    def cancel_workflow(self, workflow_id):
        self.workflow(workflow_id)
        with self.connect() as db:
            db.execute("UPDATE workflows SET cancel_requested=1,updated_at=? WHERE id=?", (now(), workflow_id))
            self._event(db, workflow_id, "workflow_cancel_requested", {})
        for job in self.workflow_jobs(workflow_id):
            if job["state"] not in TERMINAL:
                self.cancel(job["id"])

    def workflow_events(self, workflow_id):
        self.workflow(workflow_id)
        with self.connect() as db:
            rows = db.execute("SELECT * FROM events WHERE job_id=? ORDER BY seq", (workflow_id,)).fetchall()
        return [{**dict(row), "payload": json.loads(row["payload"])} for row in rows]
