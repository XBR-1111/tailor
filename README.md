# tailor

**Refine distributed robot datasets into training-ready data.**

tailor is a data-refinement pipeline for robot learning datasets (LeRobot v2/v3 format first):

- **Accept** — verify incoming distributed datasets: format-version compliance, metadata-vs-data consistency ("does the package match the label"), transfer integrity, stats snapshot vs your training config
- **Cut** — episode-level filtering (frame-level issues escalate to episode removal to preserve action-chunk coherence)
- **Sew** — deterministic cleaning only: timestamp rebuild, stats recompute, video re-encode. No imputation
- **Re-check** — cleaned output must pass the acceptance suite again before release
- **Register** — provenance ledger: source version, dropped episodes + why, recipe hash

Declarative recipes (YAML) in git; every dataset is reproducible and auditable.

*tailor-made datasets for your training.*

## Status

Early design. Issues will track the recipe schema and the first 30 acceptance expectations.
