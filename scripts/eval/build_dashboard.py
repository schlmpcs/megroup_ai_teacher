#!/usr/bin/env python3
"""Step 3 of the accuracy eval: merge judge grades with eval_slim.json, compute
aggregate stats, and render the accuracy dashboard.

Reads every ``eval_batches/grades_*.json`` (the judge output, each a list of
``{id, score, verdict}``) plus ``eval_slim.json``, then writes:

  grades.json                  canonical merged [{id, score, verdict}] (id-sorted)
  eval_graded.json             slim records + score/verdict
  chart_data.json              aggregates (overall/by_subject/by_grade/hist/heatmap/labs)
  eval_accuracy_dashboard.html self-contained dashboard (Chart.js via CDN)

Verdict is recomputed from score so it always matches the bands:
  score >= 70 -> correct ; 40-69 -> partial ; < 40 -> incorrect
Any id present in eval_slim.json but missing from the grades (empty answer /
judge skip) defaults to score 0 / incorrect.
"""
import json
import glob
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SLIM = REPO / "eval_slim.json"
BATCH_DIR = REPO / "eval_batches"
GRADES_OUT = REPO / "grades.json"
GRADED_OUT = REPO / "eval_graded.json"
CHART_OUT = REPO / "chart_data.json"
HTML_OUT = REPO / "eval_accuracy_dashboard.html"

SUBJ_ORDER = ["Физика", "Химия", "Биология"]
GRADE_ORDER = [7, 8, 9, 10, 11]


def verdict_for(score):
    if score >= 70:
        return "correct"
    if score >= 40:
        return "partial"
    return "incorrect"


def load_grades():
    """Merge all grades_*.json (judge output). Verdict recomputed from score."""
    merged = {}
    for fp in sorted(glob.glob(str(BATCH_DIR / "grades_*.json"))):
        try:
            data = json.load(open(fp, encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            print(f"  WARN: skip {fp}: {e}")
            continue
        for g in data:
            score = max(0, min(100, int(round(float(g["score"])))))
            merged[int(g["id"])] = {"id": int(g["id"]), "score": score,
                                    "verdict": verdict_for(score)}
    return merged


def stat_block(scores):
    n = len(scores)
    if n == 0:
        return {"n": 0, "correct": 0, "partial": 0, "incorrect": 0, "avg": 0.0,
                "correct_pct": 0.0, "partial_pct": 0.0, "incorrect_pct": 0.0}
    c = sum(1 for s in scores if s >= 70)
    p = sum(1 for s in scores if 40 <= s < 70)
    i = n - c - p
    return {
        "n": n, "correct": c, "partial": p, "incorrect": i,
        "avg": round(sum(scores) / n, 1),
        "correct_pct": round(c / n * 100, 1),
        "partial_pct": round(p / n * 100, 1),
        "incorrect_pct": round(i / n * 100, 1),
    }


def main():
    slim = json.load(open(SLIM, encoding="utf-8"))
    grades = load_grades()

    missing = []
    for r in slim:
        if r["id"] not in grades:
            grades[r["id"]] = {"id": r["id"], "score": 0, "verdict": "incorrect"}
            missing.append(r["id"])
    if missing:
        print(f"  {len(missing)} ungraded ids defaulted to 0/incorrect: "
              f"{missing[:20]}{'...' if len(missing) > 20 else ''}")

    GRADES_OUT.write_text(
        json.dumps([grades[i] for i in sorted(grades)], ensure_ascii=False),
        encoding="utf-8")

    graded = [{**r, "score": grades[r["id"]]["score"],
               "verdict": grades[r["id"]]["verdict"]} for r in slim]
    GRADED_OUT.write_text(json.dumps(graded, ensure_ascii=False, indent=1),
                          encoding="utf-8")

    all_scores = [g["score"] for g in graded]
    overall = stat_block(all_scores)
    overall["lenient"] = round(overall["correct_pct"] + overall["partial_pct"] / 2, 1)

    by_subject = {s: stat_block([g["score"] for g in graded if g["subject_ru"] == s])
                  for s in SUBJ_ORDER if any(g["subject_ru"] == s for g in graded)}
    by_grade = {str(gr): stat_block([g["score"] for g in graded if g["grade"] == gr])
                for gr in GRADE_ORDER if any(g["grade"] == gr for g in graded)}

    hist = [0] * 10
    for s in all_scores:
        hist[min(s // 10, 9)] += 1

    hm_avg, hm_cpct = [], []
    for s in SUBJ_ORDER:
        row_a, row_c = [], []
        for gr in GRADE_ORDER:
            sc = [g["score"] for g in graded
                  if g["subject_ru"] == s and g["grade"] == gr]
            if sc:
                row_a.append(round(sum(sc) / len(sc), 1))
                row_c.append(round(sum(1 for x in sc if x >= 70) / len(sc) * 100, 1))
            else:
                row_a.append(None)
                row_c.append(None)
        hm_avg.append(row_a)
        hm_cpct.append(row_c)

    labs = defaultdict(list)
    for g in graded:
        labs[(g["subject_ru"], g["grade"], g["lab"])].append(g["score"])
    lab_rows = []
    for (sru, gr, lab), sc in labs.items():
        n = len(sc)
        lab_rows.append({
            "name": f"{sru} {gr}кл №{lab}", "n": n,
            "avg": round(sum(sc) / n, 1),
            "correct_pct": round(sum(1 for x in sc if x >= 70) / n * 100, 1),
        })
    lab_rows.sort(key=lambda r: r["avg"])

    chart = {
        "overall": overall, "by_subject": by_subject, "by_grade": by_grade,
        "hist": hist,
        "heatmap": {"subjects": SUBJ_ORDER, "grades": GRADE_ORDER,
                    "avg": hm_avg, "correct_pct": hm_cpct},
        "weakest_labs": lab_rows[:10],
        "strongest_labs": list(reversed(lab_rows[-10:])),
    }
    CHART_OUT.write_text(json.dumps(chart, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    render_html(chart)
    print(f"DONE: graded {overall['n']} | correct {overall['correct']} "
          f"({overall['correct_pct']}%) | partial {overall['partial']} | "
          f"incorrect {overall['incorrect']} | avg {overall['avg']} | "
          f"lenient {overall['lenient']}%")
    print("by subject:", {k: f"{v['correct_pct']}% / avg {v['avg']}"
                          for k, v in by_subject.items()})


def render_html(D):
    o = D["overall"]
    HTML_OUT.write_text(HTML_TEMPLATE.format(
        correct_pct=o["correct_pct"], lenient=o["lenient"], avg=o["avg"],
        n=o["n"], data=json.dumps(D, ensure_ascii=False)), encoding="utf-8")


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RAG Eval Accuracy: VR Lab Assistant</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
 :root{{--bg:#0f1220;--card:#1a1f35;--ink:#e8ecf8;--mut:#9aa6c8;--ok:#34d399;--mid:#fbbf24;--bad:#f87171;--ac:#6366f1;}}
 *{{box-sizing:border-box}}
 body{{margin:0;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;background:linear-gradient(180deg,#0c0f1c,#0f1220);color:var(--ink)}}
 .wrap{{max-width:1180px;margin:0 auto;padding:32px 20px 64px}}
 h1{{font-size:26px;margin:0 0 4px}}
 .sub{{color:var(--mut);margin:0 0 28px;font-size:14px}}
 .kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:26px}}
 .kpi{{background:var(--card);border:1px solid #262c47;border-radius:14px;padding:18px}}
 .kpi .big{{font-size:32px;font-weight:700;line-height:1}}
 .kpi .lbl{{color:var(--mut);font-size:12px;margin-top:8px;text-transform:uppercase;letter-spacing:.04em}}
 .grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}
 .card{{background:var(--card);border:1px solid #262c47;border-radius:14px;padding:18px}}
 .card h3{{margin:0 0 14px;font-size:15px;font-weight:600}}
 .full{{grid-column:1/-1}}
 canvas{{max-height:340px}}
 table{{width:100%;border-collapse:collapse;font-size:13px}}
 th,td{{text-align:left;padding:7px 10px;border-bottom:1px solid #262c47}}
 th{{color:var(--mut);font-weight:600;font-size:11px;text-transform:uppercase}}
 td.num{{text-align:right;font-variant-numeric:tabular-nums}}
 .bar{{height:8px;border-radius:4px;background:#2a3152;overflow:hidden;min-width:80px}}
 .bar>i{{display:block;height:100%}}
 .foot{{color:var(--mut);font-size:12px;margin-top:24px;line-height:1.6}}
 @media(max-width:780px){{.grid{{grid-template-columns:1fr}}.kpis{{grid-template-columns:repeat(2,1fr)}}}}
</style>
</head>
<body><div class="wrap">
<h1>VR Lab AI Assistant: Answer Accuracy</h1>
<p class="sub">LLM-as-judge evaluation of {n} RAG answers vs teacher reference answers · physics / chemistry / biology · grades 7–11</p>

<div class="kpis">
 <div class="kpi"><div class="big" style="color:var(--ok)">{correct_pct}%</div><div class="lbl">Strict accuracy (correct)</div></div>
 <div class="kpi"><div class="big" style="color:var(--ac)">{lenient}%</div><div class="lbl">Lenient (correct + ½ partial)</div></div>
 <div class="kpi"><div class="big">{avg}</div><div class="lbl">Avg score / 100</div></div>
 <div class="kpi"><div class="big">{n}</div><div class="lbl">Questions graded</div></div>
</div>

<div class="grid">
 <div class="card"><h3>Verdict breakdown</h3><canvas id="donut"></canvas></div>
 <div class="card"><h3>Score distribution (0–100)</h3><canvas id="hist"></canvas></div>
 <div class="card"><h3>Accuracy by subject</h3><canvas id="subj"></canvas></div>
 <div class="card"><h3>Accuracy by grade</h3><canvas id="grade"></canvas></div>
 <div class="card full"><h3>Avg score by subject × grade</h3><canvas id="heat" style="max-height:300px"></canvas></div>
 <div class="card"><h3>Weakest labs (lowest avg)</h3><div id="weak"></div></div>
 <div class="card"><h3>Strongest labs (highest avg)</h3><div id="strong"></div></div>
</div>

<p class="foot">Strict accuracy counts only answers the judge rated ≥70/100 (verdict <b>correct</b>). Lenient gives half credit to <b>partial</b> (40–69). Grading by parallel Claude judge agents comparing each answer to its teacher-written reference.</p>
</div>

<script>
const D={data};
const OK='#34d399',MID='#fbbf24',BAD='#f87171',AC='#6366f1';
const gopt=(max=100)=>({{responsive:true,plugins:{{legend:{{display:false}}}},scales:{{y:{{beginAtZero:true,max,ticks:{{color:'#9aa6c8'}},grid:{{color:'#262c47'}}}},x:{{ticks:{{color:'#9aa6c8'}},grid:{{display:false}}}}}}}});

new Chart(donut,{{type:'doughnut',data:{{labels:['Correct','Partial','Incorrect'],datasets:[{{data:[D.overall.correct,D.overall.partial,D.overall.incorrect],backgroundColor:[OK,MID,BAD],borderColor:'#1a1f35',borderWidth:3}}]}},options:{{responsive:true,cutout:'62%',plugins:{{legend:{{position:'bottom',labels:{{color:'#e8ecf8',padding:14}}}}}}}}}});

new Chart(hist,{{type:'bar',data:{{labels:['0-9','10-19','20-29','30-39','40-49','50-59','60-69','70-79','80-89','90-100'],datasets:[{{data:D.hist,backgroundColor:D.hist.map((_,i)=>i<4?BAD:i<7?MID:OK)}}]}},options:gopt(null)}});

const subjK=Object.keys(D.by_subject);
new Chart(subj,{{type:'bar',data:{{labels:subjK,datasets:[
  {{label:'Correct %',data:subjK.map(k=>D.by_subject[k].correct_pct),backgroundColor:OK}},
  {{label:'Avg score',data:subjK.map(k=>D.by_subject[k].avg),backgroundColor:AC}}]}},
  options:{{...gopt(100),plugins:{{legend:{{display:true,labels:{{color:'#e8ecf8'}}}}}}}}}});

const grK=Object.keys(D.by_grade);
new Chart(grade,{{type:'bar',data:{{labels:grK.map(k=>k+' кл'),datasets:[
  {{label:'Correct %',data:grK.map(k=>D.by_grade[k].correct_pct),backgroundColor:OK}},
  {{label:'Avg score',data:grK.map(k=>D.by_grade[k].avg),backgroundColor:AC}}]}},
  options:{{...gopt(100),plugins:{{legend:{{display:true,labels:{{color:'#e8ecf8'}}}}}}}}}});

const hm=D.heatmap;
new Chart(heat,{{type:'bar',data:{{labels:hm.grades.map(g=>g+' кл'),datasets:hm.subjects.map((s,i)=>({{
  label:s,data:hm.avg[i],backgroundColor:[ '#6366f1','#22d3ee','#34d399'][i]}}))}},
  options:{{...gopt(100),plugins:{{legend:{{display:true,labels:{{color:'#e8ecf8'}}}}}}}}}});

function tbl(el,rows){{
  let h='<table><tr><th>Lab</th><th>n</th><th>Avg</th><th style="width:130px">Correct %</th></tr>';
  for(const r of rows){{const col=r.avg>=70?OK:r.avg>=50?MID:BAD;
    h+=`<tr><td>${{r.name}}</td><td class="num">${{r.n}}</td><td class="num">${{r.avg}}</td>`+
       `<td><div class="bar"><i style="width:${{r.correct_pct}}%;background:${{col}}"></i></div><small style="color:#9aa6c8">${{r.correct_pct}}%</small></td></tr>`;}}
  el.innerHTML=h+'</table>';
}}
tbl(weak,D.weakest_labs);
tbl(strong,D.strongest_labs);
</script>
</body></html>
"""


if __name__ == "__main__":
    main()
