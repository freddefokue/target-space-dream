# Contributing

The repository is intentionally smaller than the research archive. Changes
should preserve that separation.

Before opening a pull request:

```bash
pip install -e '.[test]'
pytest
python -m dream.cli
```

Please keep model-independent logic in `src/dream`, scientific orchestration in
`experiments`, and machine-specific launch commands outside the repository.
Do not commit datasets, checkpoints, Jacobians, Gram matrices, or complete run
directories. A new reported number should instead come with a compact summary,
its provenance, and a test for any new numerical invariant.

Root and solver claims must use independently replayed primal residuals. Do not
infer root nonexistence from an iterative solver exhausting its budget.

