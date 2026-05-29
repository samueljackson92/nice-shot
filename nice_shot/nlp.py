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

The available columns and a representative value for each are listed below.
Use ONLY the exact column names shown — do not invent or shorten names.

{column_table}

Translate the user's query into a JSON array of filter conditions.
Each condition must be an object with exactly three keys:
  "column"   — one of the exact column names listed above
  "operator" — one of: >=  <=  >  <  ==  !=  contains
  "value"    — a number or a string (for "contains")

Rules:
- Return ONLY valid JSON. No explanation, no markdown, no code fences.
- If multiple conditions are needed, include all of them in the array.
- If the query cannot be mapped to any column, return an empty array: []

Example output:
[{{"column": "ip_max", "operator": ">=", "value": 500000}}]
"""


@dataclass
class FilterCondition:
    column: str
    operator: str
    value: float | str


def _build_column_table(columns: list[str], sample: dict) -> str:
    """Format columns with their sample values for the system prompt."""
    lines = []
    for col in columns:
        val = sample.get(col)
        if val is None:
            lines.append(f"  {col}")
        elif isinstance(val, float):
            lines.append(f"  {col} (e.g. {val:.4g})")
        else:
            lines.append(f"  {col} (e.g. {val})")
    return "\n".join(lines)


def _build_prompt(query: str, columns: list[str], sample: dict) -> tuple[str, str]:
    column_table = _build_column_table(columns, sample)
    system = _SYSTEM_PROMPT.format(column_table=column_table)
    return system, query


def _resolve_column(name: str, available: list[str]) -> str | None:
    """Return the best matching column name from *available* for *name*.

    Resolution order:
    1. Exact match
    2. Case-insensitive exact match
    3. Unique starts-with match  (e.g. "pnbi_max" → "pnbi_max_ss")
    4. Unique contains match     (e.g. "nbi_ss"   → "pnbi_max_ss")
    """
    if name in available:
        return name

    lower_map = {c.lower(): c for c in available}
    if name.lower() in lower_map:
        return lower_map[name.lower()]

    prefix = [c for c in available if c.lower().startswith(name.lower())]
    if len(prefix) == 1:
        log.info("[NLP] resolved '%s' → '%s' (prefix match)", name, prefix[0])
        return prefix[0]

    contains = [c for c in available if name.lower() in c.lower()]
    if len(contains) == 1:
        log.info("[NLP] resolved '%s' → '%s' (contains match)", name, contains[0])
        return contains[0]

    log.warning("[NLP] could not resolve column '%s' — skipping", name)
    return None


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
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode())
            return body["message"]["content"]
    except urllib.error.URLError as exc:
        raise ConnectionError(f"Could not reach Ollama at {host}: {exc}") from exc


def _parse_conditions(raw: str) -> list[FilterCondition]:
    """Extract and validate a JSON array of filter conditions from the LLM response."""
    text = raw.strip()
    # Strip markdown fences if the model added them despite being told not to.
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(ln for ln in lines if not ln.startswith("```"))

    # Extract the first JSON array if the model wrapped it in prose.
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]

    try:
        items = json.loads(text)
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


def apply_conditions(df: pd.DataFrame, conditions: list[FilterCondition]) -> tuple[list[int], list[FilterCondition]]:
    """Apply *conditions* to *df* with fuzzy column resolution.

    Returns *(shot_ids, resolved_conditions)* where *resolved_conditions*
    reflects the actual column names used after resolution.
    """
    if not conditions:
        return [], []

    available = list(df.columns)
    resolved: list[FilterCondition] = []
    masks = []

    for cond in conditions:
        real_col = _resolve_column(cond.column, available)
        if real_col is None:
            continue
        cond_resolved = FilterCondition(column=real_col, operator=cond.operator, value=cond.value)
        resolved.append(cond_resolved)

        s = df[real_col]
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
            log.warning("NLP search: error applying condition %s: %s", cond_resolved, exc)

    if not masks:
        return [], resolved

    mask = masks[0]
    for m in masks[1:]:
        mask = mask & m

    return df.loc[mask, "shot_id"].tolist(), resolved


def search(
    query: str,
    df: pd.DataFrame,
    columns: list[str],
    host: str,
    model: str,
) -> tuple[list[int], list[FilterCondition], str]:
    """Translate *query* via Ollama and return *(shot_ids, conditions, raw_response)*.

    Raises :exc:`ConnectionError` if Ollama is unreachable, or
    :exc:`ValueError` if the LLM response cannot be parsed.
    """
    sample = df[columns].dropna().iloc[0].to_dict() if not df[columns].dropna().empty else {}
    system, user = _build_prompt(query, columns, sample)
    raw = _call_ollama(system, user, host, model)
    log.info("[NLP] raw LLM response: %s", raw[:400])
    conditions = _parse_conditions(raw)
    shot_ids, resolved = apply_conditions(df, conditions)
    return shot_ids, resolved, raw
