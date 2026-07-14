#!/usr/bin/env python3
"""Step 1 of the accuracy eval: run the reference question set against a live
``/ask`` endpoint and emit the grading inputs.

Parses ``test_questions.md`` (735 teacher Q+reference-answer pairs across
physics/chemistry/biology, grades 7-11), POSTs each question to ``/ask`` with
the matching lab context, then writes:

  eval_results.json   raw {record, result} per question (answers + citations)
  eval_results.md     human-readable Expected-vs-LLM-answer report
  eval_slim.json      [{id, subject, subject_ru, grade, lab, q, expected, answer}]
  eval_batches/batch_*.json   15-question slices ({id,q,expected,answer}) for judges

Config via env (the endpoint is a *remote* deploy, so the key must match that
server's INTERNAL_API_KEY; do not hardcode it):

  EVAL_BASE_URL   e.g. http://megroup-b560m-hdv-m-2:8001   (required)
  EVAL_API_KEY    the server's INTERNAL_API_KEY            (required)
  EVAL_WORKERS    concurrent requests (default 4)
"""
import json
import os
import re
import sys
import time
import shutil
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "test_questions.md"
OUT = REPO / "eval_results.md"
JSON_OUT = REPO / "eval_results.json"
SLIM_OUT = REPO / "eval_slim.json"
BATCH_DIR = REPO / "eval_batches"

BASE_URL = os.environ.get("EVAL_BASE_URL", "").rstrip("/")
API_KEY = os.environ.get("EVAL_API_KEY", "")
MAX_WORKERS = int(os.environ.get("EVAL_WORKERS", "4"))
BATCH_SIZE = 15

SUBJECT_MAP = {"Физика": "physics", "Химия": "chemistry", "Биология": "biology"}

subject_re = re.compile(r"^(Физика|Химия|Биология)\s*$")
grade_re = re.compile(r"^(Физика|Химия|Биология)\s+(\d+)\s+класс\s*$")
lab_re = re.compile(r"^Лабораторная работа\s+№\s*(\d+)\s*[\u2014\-–]+\s*(.*)$")
q_re = re.compile(r"^(\d+)\s*\.\s*(.*)$")
otvet_re = re.compile(r"^Ответ:\s*")


def parse(text):
    records = []
    subject_ru = subject = grade = lab_num = lab_title = None
    cur = None

    def flush():
        nonlocal cur
        if cur is not None:
            cur["expected"] = cur["expected"].strip()
            records.append(cur)
            cur = None

    for raw in text.splitlines():
        m = grade_re.match(raw)
        if m:
            flush()
            subject_ru = m.group(1)
            subject = SUBJECT_MAP[subject_ru]
            grade = int(m.group(2))
            continue
        m = subject_re.match(raw)
        if m:
            flush()
            subject_ru = m.group(1)
            subject = SUBJECT_MAP[subject_ru]
            continue
        m = lab_re.match(raw)
        if m:
            flush()
            lab_num = int(m.group(1))
            lab_title = m.group(2).strip()
            continue
        m = q_re.match(raw)
        if m:
            flush()
            cur = {
                "subject_ru": subject_ru,
                "subject": subject,
                "grade": grade,
                "lab_number": lab_num,
                "lab_title": lab_title,
                "qnum": int(m.group(1)),
                "question": m.group(2).strip(),
                "expected": "",
            }
            continue
        if cur is not None:
            line = otvet_re.sub("", raw.strip())
            if line:
                cur["expected"] += (" " if cur["expected"] else "") + line
    flush()
    return records


def ask(rec, timeout=180):
    lab = {"subject": rec["subject"], "grade": rec["grade"], "lang": "ru"}
    n = rec["lab_number"]
    if n is not None and 1 <= n <= 20:
        lab["lab_number"] = n
    body = {"query": rec["question"], "lab": lab}
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        BASE_URL + "/ask",
        data=data,
        method="POST",
        headers={
            "Authorization": "Bearer " + API_KEY,
            "Content-Type": "application/json",
        },
    )
    last_err = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            errbody = e.read().decode("utf-8", "replace")
            if e.code == 429 or e.code >= 500:
                last_err = f"HTTP {e.code}: {errbody[:200]}"
                time.sleep(min(2 ** attempt + 1, 30))
                continue
            return {"error": f"HTTP {e.code}: {errbody[:500]}"}
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(min(2 ** attempt + 1, 30))
            continue
    return {"error": last_err or "unknown error"}


def run(records):
    results = [None] * len(records)
    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        fut_to_i = {ex.submit(ask, rec): i for i, rec in enumerate(records)}
        for fut in as_completed(fut_to_i):
            i = fut_to_i[fut]
            try:
                results[i] = fut.result()
            except Exception as e:  # noqa: BLE001
                results[i] = {"error": f"future failed: {e}"}
            done += 1
            if done % 10 == 0 or done == len(records):
                rate = done / max(time.time() - t0, 1e-9) * 60
                print(f"  {done}/{len(records)} done ({rate:.0f}/min)",
                      file=sys.stderr, flush=True)
    return results


def retry_errors(records, results):
    err_idx = [i for i, r in enumerate(results) if r is None or "error" in r]
    if not err_idx:
        return
    print(f"Retrying {len(err_idx)} failed questions sequentially...",
          file=sys.stderr, flush=True)
    for n, i in enumerate(err_idx, 1):
        res = ask(records[i])
        results[i] = res
        ok = not (res is None or "error" in res)
        print(f"  [{n}/{len(err_idx)}] idx {i}: {'OK' if ok else 'STILL ERROR'}",
              file=sys.stderr, flush=True)
        time.sleep(1.5)


def render(records, results):
    errors = sum(1 for r in results if r is None or "error" in r)
    L = ["# Eval Results: VR AI Assistant\n"]
    L.append("- **Source:** `test_questions.md`")
    L.append(f"- **Endpoint:** `POST /ask` @ `{BASE_URL}`")
    L.append(f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"- **Questions:** {len(records)}  |  **Errors:** {errors}\n")
    L.append("Each question was sent with lab context (`subject`, `grade`, "
             "`lang=ru`, and `lab_number` when ≤20). Compare **Expected** "
             "against **LLM answer**.\n")
    cur_grade_key = cur_lab_key = None
    for rec, res in zip(records, results):
        gk = (rec["subject_ru"], rec["grade"])
        if gk != cur_grade_key:
            cur_grade_key, cur_lab_key = gk, None
            L.append(f"\n## {rec['subject_ru']}: {rec['grade']} класс")
        lk = (rec["lab_number"], rec["lab_title"])
        if lk != cur_lab_key:
            cur_lab_key = lk
            L.append(f"\n### Лабораторная работа №{rec['lab_number']}: {rec['lab_title']}")
            sent = f"subject={rec['subject']}, grade={rec['grade']}, lang=ru"
            n = rec["lab_number"]
            sent += f", lab_number={n}" if (n and 1 <= n <= 20) else " (lab_number omitted)"
            L.append(f"*lab context sent: {sent}*")
        L.append(f"\n#### Q{rec['qnum']}. {rec['question']}\n")
        L.append("**Expected:**\n")
        L.append(rec["expected"] or "_(empty)_")
        L.append("\n**LLM answer:**\n")
        if res is None:
            L.append("_ERROR: no result_")
        elif "error" in res:
            L.append(f"_ERROR: {res['error']}_")
        else:
            L.append(res.get("answer", "") or "_(empty answer)_")
            cits = res.get("citations") or []
            if cits:
                names = ", ".join(c.get("filename", "?") for c in cits if isinstance(c, dict))
                L.append(f"\n<sub>citations: {names}</sub>")
        L.append("\n---")
    return "\n".join(L) + "\n"


def write_slim_and_batches(records, results):
    slim = []
    for i, (rec, res) in enumerate(zip(records, results)):
        ans = "" if (res is None or "error" in res) else (res.get("answer", "") or "")
        slim.append({
            "id": i, "subject": rec["subject"], "subject_ru": rec["subject_ru"],
            "grade": rec["grade"], "lab": rec["lab_number"], "q": rec["question"],
            "expected": rec["expected"], "answer": ans,
        })
    SLIM_OUT.write_text(json.dumps(slim, ensure_ascii=False), encoding="utf-8")

    if BATCH_DIR.exists():
        shutil.rmtree(BATCH_DIR)
    BATCH_DIR.mkdir(parents=True)
    nb = 0
    for b, start in enumerate(range(0, len(slim), BATCH_SIZE)):
        chunk = slim[start:start + BATCH_SIZE]
        batch = [{"id": r["id"], "q": r["q"], "expected": r["expected"],
                  "answer": r["answer"]} for r in chunk]
        (BATCH_DIR / f"batch_{b:03d}.json").write_text(
            json.dumps(batch, ensure_ascii=False, indent=1), encoding="utf-8")
        nb += 1
    return len(slim), nb


def main():
    if not BASE_URL or not API_KEY:
        sys.exit("Set EVAL_BASE_URL and EVAL_API_KEY (the deploy's INTERNAL_API_KEY).")
    records = parse(SRC.read_text(encoding="utf-8"))
    print(f"Parsed {len(records)} questions.", file=sys.stderr, flush=True)
    print(f"  by subject: {dict(Counter(r['subject_ru'] for r in records))}",
          file=sys.stderr, flush=True)

    results = run(records)
    retry_errors(records, results)

    JSON_OUT.write_text(json.dumps(
        [{"record": rec, "result": res} for rec, res in zip(records, results)],
        ensure_ascii=False, indent=2), encoding="utf-8")
    OUT.write_text(render(records, results), encoding="utf-8")
    n_slim, n_batch = write_slim_and_batches(records, results)

    errors = sum(1 for r in results if r is None or "error" in r)
    print(f"DONE: {len(records)} questions, {errors} errors. "
          f"Wrote eval_results.json/.md, eval_slim.json ({n_slim}), {n_batch} batches.",
          file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
