# Quality Assurance & Testing

- **Unit tests** with `pytest` for data transforms and pipelines.
- **Property-based tests** with `hypothesis` for robustness (e.g., date ordering, idempotent cleaning).
- **Static checks**: ruff (lint), black (format).
- **CI** runs lint & tests on every PR/commit to `main`.
- **Data contracts**: schema checks (non-null, allowed categories) in `tests/`.
- **Reproducibility**: random seeds, pinned dependencies in `requirements.txt`.
- **Quality & testing**: schema checks, unit & property tests, reproducibility (seeds), pre-commit hooks, doc pages for QA and ethics.
All of the above is pre-wired in the repo so you can demonstrate collaborative, production-ready habits.
- **We use a CI wrapper so devs and CI run the same steps. We gate PRs with smoke tests—tiny, end-to-end checks that a synthetic dataset runs through our pipeline and produces key artefacts. It keeps feedback fast and dependable; deeper tests run nightly.**