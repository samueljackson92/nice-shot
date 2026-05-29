"""
NLP-based shot search via a local Ollama-compatible LLM.

The LLM is asked to translate a plain-English query into a JSON array of
filter conditions. Those conditions are applied to the global DataFrame to
produce a list of matching shot IDs.

No additional Python dependencies are required — HTTP calls use the
standard-library ``urllib.request``.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass

import pandas as pd

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a filter assistant for a tokamak plasma physics shot database.
The database has these numeric columns: {columns}.
A sample row is: {sample}.

Translate the user's query into a JSON array of filter conditions. Each
condition must be an object with exactly three keys:
  "column"   — one of the column names listed above
  "operator" — one of: >=  <=  >  <  ==  !=  contains
  "value"    — a number or a string (for "contains")

Return ONLY valid JSON. No explanation, no markdown, no code fences.
Example: [{{"column": "ip_max", "operator": ">=", "value": 500000}}]
If the query cannot be mapped to any column, return an empty array: []
"""


@dataclass
class FilterCondition:
    column: str
    operator: str
    value: float | str


def _build_prompt(query: str, columns: list[str], sample: dict) -> tuple[str, str]:
    system = _SYSTEM_PROMPT.format(
        columns=", ".join(columns),
        sample=json.dumps({k: v for k, v in list(sample.items())[:20]}, default=str),
    )
    return system, query


def _call_ollama(system: str, user: str, host: str, model: str) -> str:
    """POST to Ollama /api/chat and return the assistant message content."""
    payload = json.dumps(
        {
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
    ).encode()

    url = host.rstrip("/") + "/api/chat"
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode())
            return body["message"]["content"]
    except urllib.error.URLError as exc:
        raise ConnectionError(f"Could not reach Ollama at {host}: {exc}") from exc


def _parse_conditions(raw: str) -> list[FilterCondition]:
    """Extract and validate a JSON array of filter conditions from the LLM response."""
    raw = raw.strip()
    # Strip markdown fences if the model added them despite being told not to.
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(ln for ln in lines if not ln.startswith("```"))

    try:
        items = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned non-JSON: {raw!r}") from exc

    if not isinstance(items, list):
        raise ValueError(f"Expected a JSON array, got: {type(items)}")

    conditions: list[FilterCondition] = []
    valid_ops = {">=", "<=", ">", "<", "==", "!=", "contains"}
    for item in items:
        col = str(item.get("column", ""))
        op = str(item.get("operator", ""))
        val = item.get("value")
        if not col or op not in valid_ops or val is None:
            log.warning("Skipping invalid filter condition: %s", item)
            continue
        conditions.append(FilterCondition(column=col, operator=op, value=val))
    return conditions


def apply_conditions(df: pd.DataFrame, conditions: list[FilterCondition]) -> list[int]:
    """Apply *conditions* to *df* and return matching ``shot_id`` values."""
    if not conditions:
        return []

    masks = []
    for cond in conditions:
        if cond.column not in df.columns:
            log.warning("NLP search: column '%s' not in DataFrame — skipping", cond.column)
            continue
        s = df[cond.column]
        try:
            v: float | str = float(cond.value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            v = str(cond.value)

        try:
            if cond.operator == ">=":
                masks.append(s >= v)
            elif cond.operator == "<=":
                masks.append(s <= v)
            elif cond.operator == ">":
                masks.append(s > v)
            elif cond.operator == "<":
                masks.append(s < v)
            elif cond.operator == "==":
                masks.append(s == v)
            elif cond.operator == "!=":
                masks.append(s != v)
            elif cond.operator == "contains":
                masks.append(s.astype(str).str.contains(str(cond.value), case=False, na=False))
        except Exception as exc:
            log.warning("NLP search: error applying condition %s: %s", cond, exc)

    if not masks:
        return []

    mask = masks[0]
    for m in masks[1:]:
        mask = mask & m

    return df.loc[mask, "shot_id"].tolist()


def search(
    query: str,
    df: pd.DataFrame,
    columns: list[str],
    host: str,
    model: str,
) -> tuple[list[int], list[FilterCondition]]:
    """Translate *query* via Ollama and return (shot_ids, conditions).

    Raises :exc:`ConnectionError` if Ollama is unreachable, or
    :exc:`ValueError` if the LLM response cannot be parsed.
    """
    sample = df[columns].dropna().iloc[0].to_dict() if not df[columns].dropna().empty else {}
    system, user = _build_prompt(query, columns, sample)
    raw = _call_ollama(system, user, host, model)
    log.info("[NLP] raw LLM response: %s", raw[:200])
    conditions = _parse_conditions(raw)
    shot_ids = apply_conditions(df, conditions)
    return shot_ids, conditions
