#!/usr/bin/env bash

set -e
set -u
set -o pipefail

BRANCH="paradigm/provenance"

echo "=== New Paradigm Repository Transformation Script ==="
echo "Target branch: $BRANCH"
echo ""

# --- Create branch ---
git checkout -b "$BRANCH"

echo ""
echo "==> Creating directories…"
mkdir -p .github/workflows
mkdir -p .github/scripts
mkdir -p .provenance
mkdir -p audit

echo ""
echo "==> Writing New-Paradigm files…"

# --- Write provenance signer workflow ---
cat > .github/workflows/provenance-sign.yml <<'EOF'
name: Provenance Sign & Attest

on:
  push:
    branches: [ main, 'paradigm/*', 'feature/*' ]
  workflow_dispatch:

permissions:
  contents: read
  id-token: write

jobs:
  provenance:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Compute artifact hashes
        run: |
          mkdir -p .provenance
          set -o pipefail
          # Hash all tracked files except .git and .provenance and workflow yaml files
          find . -type f -not -path '*/.git/*' -not -path './.provenance/*' -not -path './.github/workflows/*' \
            -exec sha256sum {} \; | sort > .provenance/artifact.hash
          echo "Wrote .provenance/artifact.hash"

      - name: Create PROVENANCE.json
        run: |
          cat > .provenance/PROVENANCE.json <<'EOF'
{
  "project_id": "${{ github.repository }}",
  "commit_sha": "${{ github.sha }}",
  "actor": "${{ github.actor }}",
  "ref": "${{ github.ref }}",
  "timestamp": "$(date -u +%FT%TZ)",
  "artifacts_hash_file": ".provenance/artifact.hash"
}
EOF
          echo "Wrote .provenance/PROVENANCE.json"

      - name: Sign/Attest PROVENANCE (external optional)
        env:
          PROV_API_TOKEN: ${{ secrets.PROV_API_TOKEN }}
        run: |
          if [ -z "$PROV_API_TOKEN" ]; then
            echo "PROV_API_TOKEN not provided — skipping external attestation (local/dev mode)."
            cp .provenance/PROVENANCE.json .provenance/PROVENANCE.signed.json
            echo "Copied PROVENANCE.json to PROVENANCE.signed.json (unsigned placeholder)."
            exit 0
          fi

          # If you have an attestation endpoint, adapt the curl call below.
          # Default behaviour: post to provenance.example (placeholder) and save response.
          set -o pipefail
          RESPONSE=$(curl -s -X POST "https://provenance.example/api/sign" \
            -H "Authorization: Bearer $PROV_API_TOKEN" \
            -F "provenance=@.provenance/PROVENANCE.json" || true)
          if [ -z "$RESPONSE" ]; then
            echo "External attestation returned empty; storing unsigned file as fallback."
            cp .provenance/PROVENANCE.json .provenance/PROVENANCE.signed.json
          else
            echo "$RESPONSE" > .provenance/PROVENANCE.signed.json
            echo "Saved .provenance/PROVENANCE.signed.json"
          fi

      - name: Upload provenance artifact
        uses: actions/upload-artifact@v4
        with:
          name: provenance
          path: .provenance/PROVENANCE.signed.json

EOF

# --- Write provenance checker workflow ---
cat > .github/workflows/provenance-check.yml <<'EOF'
name: Provenance Check

on:
  pull_request:
    types: [opened, synchronize, reopened]

jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Try to download provenance artifact (if produced by push)
        uses: actions/download-artifact@v4
        with:
          name: provenance
        continue-on-error: true

      - name: Verify provenance file present
        run: |
          if [ ! -f ".provenance/PROVENANCE.signed.json" ]; then
            echo "::error ::Missing .provenance/PROVENANCE.signed.json; provenance required for merges."
            exit 1
          fi
          echo "Provenance file found."

      - name: Run provenance verifier
        run: |
          if ! python3 .github/scripts/verify_provenance.py .provenance/PROVENANCE.signed.json; then
            echo "::error ::Provenance verification failed."
            exit 1
          fi

EOF

# --- Write provenance verifier script ---
cat > .github/scripts/verify_provenance.py <<'EOF'
#!/usr/bin/env python3
# .github/scripts/verify_provenance.py
"""
Lightweight provenance verifier.
Checks presence of required keys and referenced artifact hash file.
Extend this to add cryptographic verification if you have signatures and public keys.
"""

import json
import sys
import os

def basic_check(path):
    if not os.path.exists(path):
        print("Provenance file not found:", path)
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            j = json.load(f)
    except Exception as e:
        print("Failed to parse provenance JSON:", e)
        return False

    required = ["project_id","commit_sha","actor","timestamp","artifacts_hash_file"]
    missing = [k for k in required if k not in j]
    if missing:
        print("Missing required keys in provenance:", missing)
        return False

    ah = j.get("artifacts_hash_file")
    if not ah or not os.path.exists(ah):
        print("Referenced artifacts hash file missing:", ah)
        return False

    # Basic sanity checks (commit sha length, timestamp format approx)
    if not isinstance(j.get("commit_sha"), str) or len(j.get("commit_sha")) < 7:
        print("Provenance commit_sha looks invalid:", j.get("commit_sha"))
        return False

    print("Provenance basic check passed.")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: verify_provenance.py PROVENANCE_FILE")
        sys.exit(2)
    if not basic_check(sys.argv[1]):
        sys.exit(1)

EOF
chmod +x .github/scripts/verify_provenance.py

# --- Architecture manifest ---
cat > ARCHITECTURE_MANIFEST.json <<'EOF'
{
  "project_id": "albiemc1303/VISION_VAE_OLM_3L_PCC_PREDICTION",
  "paradigm_standard_version": "1.0",
  "architecture_role": "OLM pipeline - frozen VAE + 3-layer LSTM predictor - real-time video latent prediction",
  "forward_stream_definitions": {
    "inputs": ["camera_frames", "vae_latents"],
    "abilities": ["latent_extraction", "temporal_aggregation", "compression", "prediction", "visualization"]
  },
  "meta_layer_rules": {
    "meta_layer_has_write_access_to_core_architecture": false,
    "meta_layer_can_call_training_loops": true,
    "meta_layer_can_modify_model_weights": true,
    "meta_layer_can_modify_workflow_files": false
  },
  "manifests": [
    "META_LAYER_RULES.md",
    "SYSTEM_BOUNDARY_CONSTRAINTS.json"
  ],
  "provenance_requirement": "Every push must produce .provenance/PROVENANCE.signed.json (CI attestation or unsigned fallback)."
}

EOF

# --- Meta layer rules ---
cat > META_LAYER_RULES.md <<'EOF'
# META_LAYER_RULES

Project: VISION_VAE_OLM_3L_PCC_PREDICTION

This file declares the meta-layer restrictions and permissions for this project,
according to the New-Paradigm living repository standard.

Rules (human-readable):

- meta_layer_has_write_access_to_core_architecture: **false**
  - Agents (programmatic drivers, bots, or models) must not modify core architecture files at runtime.
  - Core architecture files include: `src/`, `vae_processor.py`, `lstm_models.py`, `.github/`, `ARCHITECTURE_MANIFEST.json`, `META_LAYER_RULES.md`.

- meta_layer_can_call_training_loops: **true**
  - Agents may request training or evaluation jobs via approved CI runners; these jobs may update model checkpoints under `checkpoints/`.

- meta_layer_can_modify_model_weights: **true**
  - Trained models and checkpoints (e.g., `checkpoints/`) may be updated by controlled training runs.

- meta_layer_can_modify_workflow_files: **false**
  - CI/workflow changes must be performed by human maintainers and merged via PR with provenance verification.

- required_approval_for_architecture_change:
  - Maintainers must approve any changes to architecture files. Default policy: 2 human maintainers + passing provenance checks + signed release.

- external_networking_policy:
  - If agents perform network requests, the requests must be documented in commit provenance and allowed domains declared in SYSTEM_BOUNDARY_CONSTRAINTS.json.

Contact: project maintainers for policy exceptions.

EOF

# --- Boundary constraints ---
cat > SYSTEM_BOUNDARY_CONSTRAINTS.json <<'EOF'
{
  "max_runtime_accessible_paths": ["checkpoints/", "logs/", "data/"],
  "immutable_paths": [
    "src/",
    "vae_processor.py",
    "lstm_models.py",
    ".github/",
    "ARCHITECTURE_MANIFEST.json",
    "META_LAYER_RULES.md"
  ],
  "networking_constraints": {
    "external_requests_allowed": true,
    "allowed_domains": ["huggingface.co", "provenance.example"],
    "outbound_exceptions_must_be_cited": true
  },
  "resource_constraints": {
    "max_training_hours_per_job": 72,
    "max_gpu_memory_gb": 48
  }
}

EOF

# --- Basic provenance placeholder ---
touch .provenance/.gitkeep

# --- Audit script placeholder ---
cat > audit/report_audit.py <<'EOF'
#!/usr/bin/env python3
# audit/report_audit.py
"""
Audit tool to compare before and after metrics JSONs and produce a compact report.
Usage:
  python3 audit/report_audit.py before_metrics.json after_metrics.json
Output:
  - audit/audit_report.json
  - printed comparison table
"""

import json
import sys
from pathlib import Path
from pprint import pprint

def load_metrics(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def compare(before, after):
    keys = sorted(set(before.keys()) | set(after.keys()))
    table = {}
    for k in keys:
        b = before.get(k)
        a = after.get(k)
        delta = None
        try:
            if isinstance(b, (int,float)) and isinstance(a, (int,float)):
                delta = a - b
        except Exception:
            pass
        table[k] = {"before": b, "after": a, "delta": delta}
    return table

def main():
    if len(sys.argv) < 3:
        print("Usage: report_audit.py before_metrics.json after_metrics.json")
        sys.exit(2)
    before = load_metrics(sys.argv[1])
    after = load_metrics(sys.argv[2])
    table = compare(before, after)
    pprint(table)
    out = {"project": "VISION_VAE_OLM_3L_PCC_PREDICTION", "comparison": table}
    Path("audit").mkdir(parents=True, exist_ok=True)
    with open("audit/audit_report.json","w",encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Wrote audit/audit_report.json")

if __name__ == "__main__":
    main()

EOF
chmod +x audit/report_audit.py

echo ""
echo "==> Adding and committing files…"
git add .github .provenance ARCHITECTURE_MANIFEST.json META_LAYER_RULES.md SYSTEM_BOUNDARY_CONSTRAINTS.json audit
git commit -m "chore: apply New-Paradigm provenance & architecture standard"

echo ""
echo "=== COMPLETE ==="
echo "Push the branch using:"
echo "   git push origin $BRANCH"