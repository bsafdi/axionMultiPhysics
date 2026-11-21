"""The failure payload of an MCP tool, shared by the server and by the worker that runs its long jobs."""

from __future__ import annotations

import json
import re


def _field(e: BaseException, name: str, kind: type):
    """The attribute ``name`` of ``e`` if it is a non-empty ``kind``, else None.  An unrelated exception may have an
    attribute of that name with another type, or one that raises when it is read."""
    try:
        value = getattr(e, name, None)
    except Exception:  # noqa: BLE001 - a report of a failure must not fail
        return None
    return value if value and isinstance(value, kind) else None


def error_payload(e: BaseException) -> dict:
    """``ok``, ``error`` (the whole message), ``where``, ``hint``, and ``code`` and ``details`` when the error has
    them.  ``where`` and ``hint`` come from the exception's fields when they are non-empty and otherwise from the
    message text, so an exception raised from Python reads the same way.  ``code`` is used only if it is a string
    and ``details`` only if it is a dict that serializes as strict JSON: an unrelated exception may have attributes of those names.  Does not
    raise: an exception whose text or attributes cannot be read is reported without them."""
    try:
        msg = str(e)
    except Exception:  # noqa: BLE001 - a report of a failure must not fail
        msg = type(e).__name__
    out = {"ok": False, "error": msg}
    where = _field(e, "where", str)
    if where:
        out["where"] = where
    else:
        m = re.search(r"\[at ([^\]]*)\]", msg)
        if m:
            out["where"] = m.group(1)
    hint = _field(e, "hint", str)
    if hint:
        out["hint"] = hint
    elif "Hint:" in msg:
        out["hint"] = msg.split("Hint:", 1)[1].strip()
    code = _field(e, "code", str)
    if code:
        out["code"] = code
    details = _field(e, "details", dict)
    if details:
        try:
            json.dumps(details, allow_nan=False)
        except (TypeError, ValueError, RecursionError):  # not a strict JSON reply: leave it out, as the C++ side does
            details = None
        if details:
            out["details"] = details
    return out
