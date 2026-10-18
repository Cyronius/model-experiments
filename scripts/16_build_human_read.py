"""Build a blind 100-pair sheet (v2b-8L vs v1.5) for a human read, plus the answer key (2026-10-18)."""

import json
import random
from pathlib import Path

OUT = Path("data/eval/human_read")
rows = [json.loads(line) for line in Path("data/eval/eval_set.jsonl").open(encoding="utf-8")]
v2 = json.loads(Path("data/eval/titles_v2b_8L.json").read_text(encoding="utf-8"))
v15 = json.loads(Path("data/eval/titles_v15.json").read_text(encoding="utf-8"))
differ = [r for r in rows if v2[r["id"]] != v15[r["id"]]]
rng = random.Random(18)
hard = [r for r in differ if r["bucket"] == "hard"]
rest = [r for r in differ if r["bucket"] != "hard"]
picked = rng.sample(rest, 85) + rng.sample(hard, 15)
rng.shuffle(picked)

items, key = [], {}
for n, r in enumerate(picked):
    v2_first = rng.random() < 0.5
    a, b = (v2[r["id"]], v15[r["id"]]) if v2_first else (v15[r["id"]], v2[r["id"]])
    items.append({"n": n, "prompt": r["prompt"][:1500], "a": a, "b": b})
    key[n] = {"id": r["id"], "bucket": r["bucket"], "a_is": "v2" if v2_first else "v1.5"}
(OUT / "key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")

page = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Title pairs</title>
<style>
body{font:16px/1.5 system-ui,sans-serif;max-width:760px;margin:0 auto;padding:16px;background:#fafafa;color:#111}
@media(prefers-color-scheme:dark){body{background:#161616;color:#eee}pre,.card{background:#222!important;border-color:#444!important}}
pre{white-space:pre-wrap;word-break:break-word;background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px;max-height:280px;overflow:auto}
.titles{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:16px 0}
.card{border:2px solid #ccc;border-radius:10px;padding:14px;font-size:18px;background:#fff;cursor:pointer;user-select:none}.card:hover{border-color:#2563eb}.card.sel{border-color:#2563eb;background:#dbeafe!important;color:#111}
.btns{display:flex;gap:8px;flex-wrap:wrap}button{font:inherit;padding:10px 18px;border-radius:8px;border:1px solid #888;cursor:pointer}
button.sel{background:#2563eb;color:#fff;border-color:#2563eb}small{color:#888}
@media(max-width:560px){.titles{grid-template-columns:1fr}}
</style>
<h3>Which title is better? <small id=prog></small></h3>
<p><small>Judge: would a reader scanning a sidebar know what this chat is about? Click the better title, or Tie / Both bad / Reject prompt. Keys: A, T, B, X, R, arrows.</small></p>
<pre id=prompt></pre>
<div class=titles><div class=card id=cA onclick="vote('A')"><small>A (click if better)</small><br><span id=a></span></div><div class=card id=cB onclick="vote('B')"><small>B (click if better)</small><br><span id=b></span></div></div>
<div class=btns><button id=bT onclick=vote('T')>Tie</button><button id=bX onclick=vote('X')>Both bad</button><button id=bR onclick=vote('R')>Reject prompt</button>
<button onclick=go(-1)>Back</button><button onclick=go(1)>Skip</button><button onclick=exp()>Download results</button></div>
<script>
const items=__ITEMS__;let i=0;let votes={};
try{votes=JSON.parse(localStorage.getItem('votes')||'{}')}catch(e){}
function show(){const it=items[i];document.getElementById('prompt').textContent=it.prompt;
document.getElementById('a').textContent=it.a;document.getElementById('b').textContent=it.b;
const done=Object.keys(votes).length;document.getElementById('prog').textContent=`${i+1}/${items.length}  (${done} answered)`;
for(const k of ['T','X','R'])document.getElementById('b'+k).classList.toggle('sel',votes[it.n]===k);document.getElementById('cA').classList.toggle('sel',votes[it.n]==='A');document.getElementById('cB').classList.toggle('sel',votes[it.n]==='B')}
function save(){try{localStorage.setItem('votes',JSON.stringify(votes))}catch(e){}}
function vote(v){votes[items[i].n]=v;save();if(i<items.length-1)i++;show()}
function go(d){i=Math.max(0,Math.min(items.length-1,i+d));show()}
function exp(){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(votes)],{type:'application/json'}));a.download='human_votes.json';a.click()}
document.addEventListener('keydown',e=>{const k=e.key.toLowerCase();if(k==='a'||k==='t'||k==='b'||k==='x'||k==='r')vote(k.toUpperCase());else if(k==='arrowleft')go(-1);else if(k==='arrowright')go(1)});
show();
</script>""".replace("__ITEMS__", json.dumps(items, ensure_ascii=False).replace("</", "<\/"))
(OUT / "pairs.html").write_text(page, encoding="utf-8")
print(f"{len(items)} pairs ({sum(r['bucket'] == 'hard' for r in picked)} hard) -> {OUT / 'pairs.html'}")


def cell(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return (text[:limit] + "…" if len(text) > limit else text).replace("|", "\|")


lines = ["# Which title is better? Write A, B or T (tie) in the Pick column", "",
         "| # | Prompt (start) | Title A | Title B | Pick |", "|--:|---|---|---|:-:|"]
lines += [f"| {it['n'] + 1} | {cell(it['prompt'], 140)} | {cell(it['a'], 90)} | {cell(it['b'], 90)} |  |" for it in items]
(OUT / "pairs.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

table_page = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Title pairs table</title>
<style>
body{font:15px/1.45 system-ui,sans-serif;margin:0;background:#fafafa;color:#111}
@media(prefers-color-scheme:dark){body{background:#161616;color:#eee}th{background:#222!important}tr:nth-child(even){background:#1d1d1d!important}.bar{background:#161616!important}}
.bar{position:sticky;top:0;background:#fafafa;padding:10px 16px;border-bottom:1px solid #888;display:flex;gap:12px;align-items:center;z-index:2}
table{border-collapse:collapse;width:100%}th,td{padding:8px 10px;text-align:left;vertical-align:top;border-bottom:1px solid #8884}
th{position:sticky;top:49px;background:#eee}tr:nth-child(even){background:#f2f2f2}
.p{max-width:360px;white-space:pre-wrap;word-break:break-word;font-size:13px;color:#666}.p b{cursor:pointer;color:#2563eb;font-weight:normal}
label{margin-right:10px;cursor:pointer;white-space:nowrap}td.t{font-weight:600;cursor:pointer}td.t:hover{outline:2px solid #2563eb;outline-offset:-2px}td.t.sel{background:#dbeafe!important;color:#111;outline:2px solid #2563eb;outline-offset:-2px}
button{font:inherit;padding:6px 14px;border-radius:8px;border:1px solid #888;cursor:pointer}
button.vb{padding:5px 10px;font-size:13px}button.vb.sel{background:#2563eb;color:#fff;border-color:#2563eb}tr.rej{opacity:.45}tr.rej button.vb.sel{background:#b91c1c;border-color:#b91c1c}
td:last-child{white-space:nowrap}
</style>
<div class=bar><b id=prog></b><button onclick=exp()>Download results</button><span style="color:#888">Click the better title, or Tie / Both bad / Reject (inappropriate prompt). Answers save in this browser.</span></div>
<table><thead><tr><th>#</th><th>Prompt</th><th>Title A</th><th>Title B</th><th>Other</th></tr></thead><tbody id=rows></tbody></table>
<script>
const items=__ITEMS__;let votes={};try{votes=JSON.parse(localStorage.getItem('votes_table')||'{}')}catch(e){}
const rows=document.getElementById('rows');
function esc(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
items.forEach(it=>{const tr=document.createElement('tr');
const short=it.prompt.length>260?it.prompt.slice(0,260):it.prompt;
tr.innerHTML=`<td>${it.n+1}</td><td class=p>${esc(short)}${it.prompt.length>260?' <b onclick="this.parentNode.textContent=items[${it.n}].prompt">…more</b>':''}</td><td class='t ${votes[it.n]==='A'?'sel':''}' data-n=${it.n} data-v=A>${esc(it.a)}</td><td class='t ${votes[it.n]==='B'?'sel':''}' data-n=${it.n} data-v=B>${esc(it.b)}</td><td>`+
[['T','Tie'],['X','Both bad'],['R','Reject']].map(([v,l])=>`<button class='vb ${votes[it.n]===v?'sel':''}' data-n=${it.n} data-v=${v}>${l}</button>`).join(' ')+'</td>';
if(votes[it.n]==='R')tr.classList.add('rej');
rows.appendChild(tr)});
function setVote(n,v){votes[n]=v;try{localStorage.setItem('votes_table',JSON.stringify(votes))}catch(_){}
const tr=rows.children[n];tr.querySelectorAll('td.t,button.vb').forEach(el=>el.classList.toggle('sel',el.dataset.v===v));
tr.classList.toggle('rej',v==='R');count()}
rows.addEventListener('click',e=>{const el=e.target.closest('td.t,button.vb');if(el)setVote(el.dataset.n,el.dataset.v)});
function count(){document.getElementById('prog').textContent=Object.keys(votes).length+'/'+items.length+' answered'}
function exp(){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(votes)],{type:'application/json'}));a.download='human_votes.json';a.click()}
count();
</script>""".replace("__ITEMS__", json.dumps(items, ensure_ascii=False).replace("</", "<\/"))
(OUT / "pairs_table.html").write_text(table_page, encoding="utf-8")
