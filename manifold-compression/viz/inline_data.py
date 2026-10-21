# Inline viz_data.json into the viewer template, writing manifold_viewer.html (2026-10-18).
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
d = json.loads((HERE / "viz_data.json").read_text(encoding="utf-8"))
for s in d["sites"]:
    s["xyz"] = [[round(v, 2) for v in p] for p in s["xyz"]]
d["paths"] = [[[round(v, 2) for v in p] for p in path] for path in d["paths"]]
# ASCII-only JSON keeps byte-fragment tokens (U+FFFD) and other rare characters as \u escapes in the page source.
data = json.dumps(d, ensure_ascii=True, separators=(",", ":")).replace("</", "<\\/")
template = (HERE / "manifold_viewer.template.html").read_text(encoding="utf-8")
out = HERE / "manifold_viewer.html"
out.write_text(template.replace("/*__DATA__*/", data), encoding="utf-8")
print(out, out.stat().st_size // 1024, "KB", "�" in out.read_text(encoding="utf-8"))
