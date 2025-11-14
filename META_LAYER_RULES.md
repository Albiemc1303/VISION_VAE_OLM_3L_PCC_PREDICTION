# META_LAYER_RULES

Project: VISION_VAE_OLM_3L_PCC_PREDICTION

This file declares the meta-layer restrictions for this project per the New-Paradigm standard.

- meta_layer_has_write_access_to_core_architecture: **false**
  - Meaning: runtime control code (agents/LLMs) must not be allowed to edit or replace core system files (src/, .github/workflows/, ARCHITECTURE_MANIFEST.json, META_LAYER_RULES.md)
- meta_layer_can_call_training_loops: **true**
  - agents may request training runs, retraining, and hyperparameter experiments if done via CI jobs or approved runners
- meta_layer_can_modify_model_weights: **true**
  - training updates may modify checkpoint files (checkpoints/), but not architecture files
- meta_layer_can_modify_workflow_files: **false**
  - changes to CI/workflows must be human-reviewed and merged via PRs with provenance checks
- required_approval_for_architecture_change: "2 human maintainers + provenance-attested release"
