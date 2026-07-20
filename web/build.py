"""
web/build.py  --  (re)build the interactive demo from the exported JSON.

Pipeline:  run_demo.py  ->  outputs/integrated_demo.json  ->  web/index.html

`web/template.html` is the demo UI with a `/*DEMO_DATA*/` placeholder; this script
inlines the latest simulation data into it and writes a standalone `web/index.html`
that opens offline (no server needed -- the data is embedded, not fetched).

    python web/build.py                       # uses outputs/integrated_demo.json
    python web/build.py path/to/other.json    # or a specific data file
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def main():
    data_path = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "outputs" / "integrated_demo.json"
    if not data_path.exists():
        sys.exit(f"[build] {data_path} not found -- run `python run_demo.py` first.")

    template = (HERE / "template.html").read_text(encoding="utf-8")
    if "/*DEMO_DATA*/" not in template:
        sys.exit("[build] web/template.html is missing the /*DEMO_DATA*/ placeholder.")

    data = data_path.read_text(encoding="utf-8")
    body = template.replace("/*DEMO_DATA*/", data)
    head = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>FrugalNav - live flight console</title>'
            '<style>html,body{margin:0;background:#0a0f1a}</style></head><body>')
    out = HERE / "index.html"
    out.write_text(head + body + "</body></html>", encoding="utf-8")
    print(f"[build] wrote {out.relative_to(REPO)} ({out.stat().st_size // 1024} KB) "
          f"from {data_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
