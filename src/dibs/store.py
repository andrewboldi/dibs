"""The daemon's memory: jobs, a few settings, and guard events, in SQLite.

The database holds each submitter's environment (which can contain API keys), so the file is
private to the user. Only the daemon opens it.
"""

import json
import os
import sqlite3
from collections.abc import Iterable
from pathlib import Path

from dibs.model import TERMINAL, Job, State, job_from_dict, job_to_dict

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (id INTEGER PRIMARY KEY AUTOINCREMENT, state TEXT NOT NULL, data TEXT NOT NULL, ended REAL);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, time REAL NOT NULL, data TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS jobs_state ON jobs (state);
"""
_TERMINAL = tuple(str(s) for s in TERMINAL)


def _command(cwd: str, argv: list[str]) -> str:
    """What makes two jobs the same command: the folder and the exact command line."""
    return json.dumps([cwd, list(argv)])


class Store:
    def __init__(self, path: Path):
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        os.close(fd)
        os.chmod(path, 0o600)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        with self._db:
            self._db.executescript(_SCHEMA)
            self._add_ended_column()
            self._add_command_column()

    def _add_ended_column(self) -> None:
        """Databases from before the ended column get it, filled in from each job's data."""
        columns = {row[1] for row in self._db.execute("PRAGMA table_info(jobs)")}
        if "ended" not in columns:
            self._db.execute("ALTER TABLE jobs ADD COLUMN ended REAL")
            rows = self._db.execute("SELECT id, data FROM jobs").fetchall()
            self._db.executemany(
                "UPDATE jobs SET ended = ? WHERE id = ?", [(json.loads(data).get("ended"), i) for i, data in rows]
            )
        self._db.execute("CREATE INDEX IF NOT EXISTS jobs_ended ON jobs (ended)")

    def _add_command_column(self) -> None:
        """Databases from before the command column get it, filled in from each job's data."""
        columns = {row[1] for row in self._db.execute("PRAGMA table_info(jobs)")}
        if "command" not in columns:
            self._db.execute("ALTER TABLE jobs ADD COLUMN command TEXT")
            rows = self._db.execute("SELECT id, data FROM jobs").fetchall()
            commands = []
            for i, data in rows:
                d = json.loads(data) if data else {}
                commands.append((_command(d.get("cwd", ""), d.get("argv", [])), i))
            self._db.executemany("UPDATE jobs SET command = ? WHERE id = ?", commands)
        self._db.execute("CREATE INDEX IF NOT EXISTS jobs_command ON jobs (command)")

    def close(self) -> None:
        self._db.close()

    def add_job(self, job: Job) -> int:
        with self._db:
            cur = self._db.execute("INSERT INTO jobs (state, data) VALUES (?, '')", (str(job.state),))
            job.id = cur.lastrowid
            self._db.execute(
                "UPDATE jobs SET data = ?, ended = ?, command = ? WHERE id = ?",
                (json.dumps(job_to_dict(job, include_env=True)), job.ended, _command(job.cwd, job.argv), job.id),
            )
        return job.id

    def update_job(self, job: Job) -> None:
        with self._db:
            self._db.execute(
                "UPDATE jobs SET state = ?, data = ?, ended = ?, command = ? WHERE id = ?",
                (str(job.state), json.dumps(job_to_dict(job, include_env=True)), job.ended,
                 _command(job.cwd, job.argv), job.id),
            )  # fmt: skip

    def earlier_runs(self, cwd: str, argv: list[str], limit: int = 3) -> list[Job]:
        """Finished runs of this command in this folder, newest first."""
        marks = ",".join("?" * len(_TERMINAL))
        rows = self._db.execute(
            f"SELECT data FROM jobs WHERE command = ? AND state IN ({marks}) ORDER BY ended DESC, id DESC LIMIT ?",
            (_command(cwd, argv), *_TERMINAL, limit),
        ).fetchall()
        return [job_from_dict(json.loads(r[0])) for r in rows]

    def get_job(self, job_id: int) -> Job | None:
        row = self._db.execute("SELECT data FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return job_from_dict(json.loads(row[0])) if row else None

    def jobs(self, states: Iterable[State] | None = None) -> list[Job]:
        if states is None:
            rows = self._db.execute("SELECT data FROM jobs ORDER BY id").fetchall()
        else:
            wanted = [str(s) for s in states]
            marks = ",".join("?" * len(wanted))
            rows = self._db.execute(
                f"SELECT data FROM jobs WHERE state IN ({marks}) ORDER BY id", wanted
            ).fetchall()
        return [job_from_dict(json.loads(r[0])) for r in rows]

    def _finished(self, condition: str, t: float) -> list[Job]:
        marks = ",".join("?" * len(_TERMINAL))
        rows = self._db.execute(
            f"SELECT data FROM jobs WHERE state IN ({marks}) AND ended {condition} ? ORDER BY id",
            (*_TERMINAL, t),
        ).fetchall()
        return [job_from_dict(json.loads(r[0])) for r in rows]

    def finished_before(self, t: float) -> list[Job]:
        """Finished jobs that ended before t (found by the ended column, not by decoding every job)."""
        return self._finished("<", t)

    def finished_since(self, t: float) -> list[Job]:
        """Finished jobs that ended at or after t."""
        return self._finished(">=", t)

    def delete_jobs(self, ids: Iterable[int]) -> None:
        with self._db:
            self._db.executemany("DELETE FROM jobs WHERE id = ?", [(i,) for i in ids])

    def get_meta(self, key: str) -> str | None:
        row = self._db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._db:
            self._db.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def add_event(self, kind: str, data: dict, time: float) -> None:
        with self._db:
            self._db.execute(
                "INSERT INTO events (kind, time, data) VALUES (?, ?, ?)", (kind, time, json.dumps(data))
            )

    def events(self, since: float) -> list[dict]:
        rows = self._db.execute(
            "SELECT kind, time, data FROM events WHERE time >= ? ORDER BY id", (since,)
        ).fetchall()
        return [{"kind": kind, "time": time, **json.loads(data)} for kind, time, data in rows]

    def delete_events_before(self, t: float) -> None:
        with self._db:
            self._db.execute("DELETE FROM events WHERE time < ?", (t,))
