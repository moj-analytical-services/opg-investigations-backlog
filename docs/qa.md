# Quality Assurance & Testing

- **Unit tests** with `pytest` for data transforms and pipelines.
- **Property-based tests** with `hypothesis` for robustness (e.g., date ordering, idempotent cleaning).
- **Static checks**: ruff (lint), black (format).
- **CI** runs lint & tests on every PR/commit to `main`.
- **Data contracts**: schema checks (non-null, allowed categories) in `tests/`.
- **Reproducibility**: random seeds, pinned dependencies in `requirements.txt`.
