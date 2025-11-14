#!/usr/bin/env python3
# audit/report_audit.py
import json
import sys
import os
from pathlib import Path
from pprint import pprint

def load_metrics(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def compare(before, after):
    keys = sorted(set(before.keys()) | set(after.keys()))
    table = {}
    for k in keys:
        b = before.get(k, None)
        a = after.get(k, None)
        delta = None
        try:
            if b is not None and a is not None:
                delta = a - b
        except Exception:
            pass
        table[k] = {"before": b, "after": a, "delta": delta}
    return table

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: report_audit.py before_metrics.json after_metrics.json")
        sys.exit(2)
    before = load_metrics(sys.argv[1])
    after = load_metrics(sys.argv[2])
    table = compare(before, after)
    pprint(table)
    out = {"comparison": table}
    Path("audit").mkdir(parents=True, exist_ok=True)
    with open("audit/audit_report.json","w",encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Wrote audit/audit_report.json")
# (a script producing before/after comparison given saved metrics JSON — used for PR)