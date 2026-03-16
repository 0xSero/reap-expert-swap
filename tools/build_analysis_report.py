#!/usr/bin/env python3
"""Build a single-page tabbed HTML report from analysis/*.md files."""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import markdown

ANALYSIS_DIR = Path(__file__).resolve().parents[1] / "analysis"
OUTPUT = ANALYSIS_DIR / "report.html"

DOCS = [
    ("README.md", "Index"),
    ("01-system-architecture.md", "Architecture"),
    ("02-root-cause-analysis.md", "Root Causes"),
    ("03-scoring-pipeline-trace.md", "Scoring Trace"),
    ("04-reap-paper-comparison.md", "REAP Paper"),
    ("05-vllm-runtime-analysis.md", "vLLM Runtime"),
    ("06-budget-arithmetic.md", "Budget Math"),
    ("07-dataset-methodology-audit.md", "Datasets"),
    ("08-strict30-v2-spec.md", "v2 Spec"),
    ("09-validation-script-design.md", "Validation"),
]

MERMAID_FILES = list((ANALYSIS_DIR / "diagrams").glob("*.mermaid"))

def render_md(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    # Convert mermaid code blocks to divs for mermaid.js
    text = re.sub(
        r'```mermaid\n(.*?)```',
        r'<pre class="mermaid">\1</pre>',
        text, flags=re.DOTALL
    )
    return markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "codehilite", "toc"],
        extension_configs={"codehilite": {"css_class": "code"}},
    )

def build_html() -> str:
    tabs_html = []
    panels_html = []
    for i, (filename, label) in enumerate(DOCS):
        path = ANALYSIS_DIR / filename
        if not path.exists():
            continue
        tab_id = f"tab-{i}"
        active = " active" if i == 0 else ""
        tabs_html.append(f'<button class="tab{active}" onclick="openTab(event,\'{tab_id}\')">{label}</button>')
        content = render_md(path)
        display = "block" if i == 0 else "none"
        panels_html.append(f'<div id="{tab_id}" class="tab-panel" style="display:{display}">{content}</div>')

    # Add diagrams tab
    if MERMAID_FILES:
        tab_id = "tab-diagrams"
        tabs_html.append(f'<button class="tab" onclick="openTab(event,\'{tab_id}\')">Diagrams</button>')
        diagram_html = "<h1>Mermaid Diagrams</h1>"
        for mf in sorted(MERMAID_FILES):
            mermaid_src = mf.read_text(encoding="utf-8")
            name = mf.stem.replace("-", " ").title()
            diagram_html += f'<h2>{name}</h2><pre class="mermaid">{mermaid_src}</pre>'
        panels_html.append(f'<div id="{tab_id}" class="tab-panel" style="display:none">{diagram_html}</div>')

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>strict30 Deep Analysis</title>
<script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
<style>
:root {{
  --bg: #0d1117; --bg2: #161b22; --bg3: #21262d;
  --fg: #c9d1d9; --fg2: #8b949e; --accent: #58a6ff;
  --green: #3fb950; --red: #f85149; --yellow: #d29922; --orange: #db6d28;
  --border: #30363d; --code-bg: #1a1f27;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
       background: var(--bg); color: var(--fg); line-height: 1.6; }}
.header {{ background: var(--bg2); border-bottom: 1px solid var(--border);
           padding: 16px 24px; display: flex; align-items: center; gap: 16px; }}
.header h1 {{ font-size: 20px; font-weight: 600; }}
.header .badge {{ background: var(--red); color: #fff; padding: 2px 8px;
                  border-radius: 12px; font-size: 12px; font-weight: 600; }}
.tab-bar {{ display: flex; flex-wrap: wrap; gap: 2px; background: var(--bg2);
            padding: 4px 24px 0; border-bottom: 1px solid var(--border);
            position: sticky; top: 0; z-index: 100; }}
.tab {{ background: transparent; color: var(--fg2); border: none;
        padding: 8px 16px; cursor: pointer; font-size: 13px; font-weight: 500;
        border-bottom: 2px solid transparent; transition: all 0.15s; }}
.tab:hover {{ color: var(--fg); background: var(--bg3); }}
.tab.active {{ color: var(--accent); border-bottom-color: var(--accent); }}
.tab-panel {{ max-width: 960px; margin: 0 auto; padding: 32px 24px; }}
h1 {{ font-size: 28px; font-weight: 600; margin: 24px 0 16px; padding-bottom: 8px;
      border-bottom: 1px solid var(--border); }}
h2 {{ font-size: 22px; font-weight: 600; margin: 32px 0 12px; color: var(--accent); }}
h3 {{ font-size: 17px; font-weight: 600; margin: 24px 0 8px; }}
h4 {{ font-size: 14px; font-weight: 600; margin: 16px 0 6px; color: var(--fg2); }}
p {{ margin: 8px 0; }}
a {{ color: var(--accent); text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
ul, ol {{ padding-left: 24px; margin: 8px 0; }}
li {{ margin: 4px 0; }}
table {{ border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 14px; }}
th {{ background: var(--bg3); text-align: left; padding: 10px 12px;
      border: 1px solid var(--border); font-weight: 600; white-space: nowrap; }}
td {{ padding: 8px 12px; border: 1px solid var(--border); vertical-align: top; }}
tr:hover td {{ background: var(--bg2); }}
code {{ font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
        background: var(--code-bg); padding: 2px 6px; border-radius: 4px; font-size: 13px; }}
pre {{ background: var(--code-bg); border: 1px solid var(--border); border-radius: 8px;
       padding: 16px; overflow-x: auto; margin: 12px 0; font-size: 13px; line-height: 1.5; }}
pre code {{ background: none; padding: 0; }}
pre.mermaid {{ background: var(--bg2); border: 1px solid var(--border); border-radius: 8px;
              padding: 24px; text-align: center; }}
blockquote {{ border-left: 3px solid var(--accent); padding-left: 16px;
              color: var(--fg2); margin: 12px 0; }}
strong {{ color: #f0f6fc; }}
em {{ color: var(--fg2); }}
hr {{ border: none; border-top: 1px solid var(--border); margin: 24px 0; }}
.code {{ background: var(--code-bg); border-radius: 8px; padding: 16px;
         overflow-x: auto; margin: 12px 0; }}
.code pre {{ background: none; border: none; padding: 0; margin: 0; }}
@media (max-width: 768px) {{
  .tab {{ padding: 6px 10px; font-size: 12px; }}
  .tab-panel {{ padding: 16px; }}
  table {{ font-size: 12px; }}
}}
</style>
</head>
<body>
<div class="header">
  <h1>strict30 Deep Analysis</h1>
  <span class="badge">FAKE DYNAMIC</span>
  <span style="color:var(--fg2);font-size:13px">Qwen3.5-35B-A3B &middot; 2026-03-16</span>
</div>
<div class="tab-bar">
{''.join(tabs_html)}
</div>
{''.join(panels_html)}
<script>
function openTab(evt, id) {{
  document.querySelectorAll('.tab-panel').forEach(p => p.style.display = 'none');
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById(id).style.display = 'block';
  evt.currentTarget.classList.add('active');
  // Re-render mermaid in newly visible tab
  if (window.mermaid) {{
    mermaid.run({{ querySelector: '#' + id + ' .mermaid' }});
  }}
}}
mermaid.initialize({{ startOnLoad: true, theme: 'dark',
  themeVariables: {{ primaryColor: '#21262d', primaryBorderColor: '#58a6ff',
    primaryTextColor: '#c9d1d9', lineColor: '#58a6ff', secondaryColor: '#161b22',
    tertiaryColor: '#0d1117' }}
}});
</script>
</body>
</html>"""


if __name__ == "__main__":
    html = build_html()
    OUTPUT.write_text(html, encoding="utf-8")
    print(f"Written {len(html):,} bytes to {OUTPUT}")
