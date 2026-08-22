"""Substitute the measured sparsity bit-strings into the report placeholders."""
import re

REPORT = "docs/superpowers/plans/gpu-admm-report.html"
PATS = "/root/.claude/jobs/a1258382/tmp/patterns.txt"


def main():
    src = open(PATS).read()
    pats = dict(re.findall(r'const (\w+)_PAT = "([01,]+)";', src))
    assert set(pats) == {"P", "K"}, pats.keys()
    html = open(REPORT).read()
    for k, v in pats.items():
        placeholder = "__" + k + "_PAT__"
        assert placeholder in html, f"missing {placeholder}"
        html = html.replace(placeholder, v)
    open(REPORT, "w").write(html)
    for k, v in pats.items():
        rows = v.split(",")
        filled = sum(r.count("1") for r in rows)
        print(f"  {k}: {len(rows)} rows x {len(rows[0])} cols, {filled} filled cells")
    print("injected OK; remaining placeholders:",
          html.count("__P_PAT__") + html.count("__K_PAT__"))


if __name__ == "__main__":
    main()
