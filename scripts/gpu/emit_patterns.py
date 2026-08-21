"""Emit the measured sparsity grids as compact comma-joined bit strings,
for embedding in the findings report.
"""
import json

SRC = "/root/.claude/jobs/a1258382/tmp/sparsity.json"
OUT = "/root/.claude/jobs/a1258382/tmp/patterns.txt"


def main():
    d = json.load(open(SRC))
    with open(OUT, "w") as f:
        for name in ("P", "K"):
            rows = ["".join("1" if v else "0" for v in r) for r in d[name]]
            f.write(f'const {name}_PAT = "' + ",".join(rows) + '";\n')
    print("wrote", OUT)


if __name__ == "__main__":
    main()
