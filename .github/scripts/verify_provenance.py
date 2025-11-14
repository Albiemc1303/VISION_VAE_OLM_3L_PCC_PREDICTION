#!/usr/bin/env python3
import json
import sys
import os

def basic_check(path):
    if not os.path.exists(path):
        print("Provenance file not found:", path)
        return False
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)
    # required keys
    required = ["project_id","commit_sha","actor","timestamp","artifacts_hash_file"]
    missing = [k for k in required if k not in j]
    if missing:
        print("Missing required keys in provenance:", missing)
        return False
    ah = j.get("artifacts_hash_file")
    if not os.path.exists(ah):
        print("Artifacts hash file referenced missing:", ah)
        # allow leniency for template testing
        return False
    print("Provenance basic check passed.")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: verify_provenance.py PROVENANCE_FILE")
        sys.exit(2)
    ok = basic_check(sys.argv[1])
    sys.exit(0 if ok else 3)
# Simple local verifier — can be extended to cryptographic checks
# Make it executable chmod +x .github/scripts/verify_provenance.py.