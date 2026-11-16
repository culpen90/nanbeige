"""Local notes and a bounded history of completed chat turns."""

from __future__ import annotations

from collections.abc import Collection
import os
from pathlib import Path
import re
import sqlite3
import stat
import unicodedata


MAX_SAVED_TURNS = 256
MAX_NOTES = 512
MAX_ENTRY_CHARS = 2_000
_APPLICATION_ID = 0x4E424D31  # NBM1
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]+")
_WORDS = re.compile(r"[^\W_]+")
_STOP = frozenset(
    "about after again all also am an and any are as at be been but by can could did do does "
    "for from get had has have how i if in into is it its just me more my of on or our out "
    "please said say should so some that the their them then there these they this those to "
    "too up us want was we were what when where which who will with would you your".split()
)


def _terms(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    terms = set()
    for match in _CJK.finditer(normalized):
        run = match[0]
        terms.update(run[index:index + 2] for index in range(len(run) - 1))
        if len(run) == 1:
            terms.add(run)
    terms.update(
        word for word in _WORDS.findall(_CJK.sub(" ", normalized))
        if word not in _STOP and (len(word) >= 2 or word.isdecimal())
    )
    return terms


def _excerpt(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    # Facts and corrections can be at either end of a long message.
    head = (limit - 3) * 2 // 3
    return text[:head] + " … " + text[-(limit - 3 - head):]


def _stored_turn(prompt: str, reply: str) -> tuple[str, str]:
    return _excerpt(prompt.strip(), MAX_ENTRY_CHARS), _excerpt(reply.strip(), MAX_ENTRY_CHARS)


def _recall_excerpt(text: str, limit: int, matched: set[str]) -> str:
    excerpt = _excerpt(text, limit)
    score = len(matched & _terms(excerpt)) if matched else 0
    if len(text) <= limit or score == len(matched):
        return excerpt

    # The default excerpt preserves both ends. If the query matched the
    # middle, use the window with the most matching terms instead.
    width = limit - 6  # space for ellipses at both ends
    for start in range(0, len(text), width // 2):
        end = start + width
        window = text[start:end]
        hits = len(matched & _terms(window))
        if hits > score:
            score = hits
            excerpt = (" … " if start else "") + window + (" … " if end < len(text) else "")
            if score == len(matched):
                break
    return excerpt


class MemoryStore:
    def __init__(self, path: Path | None):
        if path is None:
            self._db = sqlite3.connect(":memory:")
        else:
            path = path.expanduser().absolute()
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            try:
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                permissions = path.lstat().st_mode
                if not stat.S_ISREG(permissions) or permissions & 0o077:
                    raise ValueError("Choose a regular memory file with owner-only permissions.") from None
            else:
                os.close(fd)
            self._db = sqlite3.connect(path.as_uri() + "?mode=rw", uri=True)

        try:
            self._db.execute("PRAGMA secure_delete = ON")
            with self._db:
                self._db.execute("BEGIN IMMEDIATE")
                (application_id,) = self._db.execute("PRAGMA application_id").fetchone()
                if application_id != _APPLICATION_ID:
                    if application_id or self._db.execute("SELECT 1 FROM sqlite_schema LIMIT 1").fetchone():
                        raise ValueError("The selected file is not a Nanbeige memory database.")
                    self._db.execute(
                        "CREATE TABLE notes (id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT NOT NULL)"
                    )
                    self._db.execute(
                        "CREATE TABLE turns (id INTEGER PRIMARY KEY, prompt TEXT NOT NULL, reply TEXT NOT NULL)"
                    )
                    self._db.execute(f"PRAGMA application_id = {_APPLICATION_ID}")
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> MemoryStore:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._db.close()

    def remember(self, text: str) -> int:
        text = text.strip()
        if not text or len(text) > MAX_ENTRY_CHARS:
            raise ValueError(f"Enter a memory note between 1 and {MAX_ENTRY_CHARS:,} characters.")
        with self._db:
            result = self._db.execute(
                "INSERT INTO notes (content) SELECT ? WHERE (SELECT COUNT(*) FROM notes) < ?",
                (text, MAX_NOTES),
            )
            if not result.rowcount:
                raise ValueError(f"Memory already has {MAX_NOTES} notes. Forget a note before adding another.")
            return result.lastrowid

    def record(self, prompt: str, reply: str) -> None:
        prompt, reply = _stored_turn(prompt, reply)
        if not prompt:
            return
        with self._db:
            self._db.execute("INSERT INTO turns (prompt, reply) VALUES (?, ?)", (prompt, reply))
            self._db.execute(
                "DELETE FROM turns WHERE id <= (SELECT id FROM turns ORDER BY id DESC LIMIT 1 OFFSET ?)",
                (MAX_SAVED_TURNS,),
            )

    def notes(self) -> list[tuple[int, str]]:
        return self._db.execute("SELECT id, content FROM notes ORDER BY id").fetchall()

    def history(self) -> list[tuple[str, str]]:
        return self._db.execute("SELECT prompt, reply FROM turns ORDER BY id").fetchall()

    def forget(self, id: int) -> bool:
        with self._db:
            return self._db.execute("DELETE FROM notes WHERE id = ?", (id,)).rowcount > 0

    def clear(self) -> None:
        with self._db:
            self._db.execute("DELETE FROM notes")
            self._db.execute("DELETE FROM turns")

    def recall(
        self, query: str, *, limit: int = 4, exclude: Collection[tuple[str, str]] = (),
    ) -> list[str]:
        if limit <= 0:
            return []
        terms = _terms(query)
        excluded = {_stored_turn(prompt, reply) for prompt, reply in exclude}
        matches: list[tuple[int, bool, int, str]] = []
        recent_note = recent_turn = None
        for note_id, note in self.notes():
            note_terms = terms & _terms(note)
            excerpt = f"Saved note: {_recall_excerpt(note, 360, note_terms)}"
            recent_note = excerpt
            if note_terms:
                matches.append((3 * len(note_terms), True, note_id, excerpt))
        for turn_id, (prompt, reply) in enumerate(self.history()):
            if (prompt, reply) in excluded:
                continue
            user_terms = terms & _terms(prompt)
            assistant_terms = terms & _terms(reply)
            excerpt = f"Previous user: {_recall_excerpt(prompt, 260, user_terms)}"
            if reply:
                excerpt += f"\nPrevious assistant: {_recall_excerpt(reply, 200, assistant_terms)}"
            recent_turn = excerpt
            score = 2 * len(user_terms) + len(assistant_terms - user_terms)
            if score:
                matches.append((score, False, turn_id, excerpt))
        matches.sort(reverse=True)
        results = [match[-1] for match in matches]
        if recent_note and not any(is_note for _, is_note, _, _ in matches):
            results.append(recent_note)
        if recent_turn and not matches:
            results.append(recent_turn)
        return list(dict.fromkeys(results))[:limit]
