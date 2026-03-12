# Contributing

## Principles

- preserve determinism where possible
- prefer artifact-backed claims over prose-only claims
- keep benchmark and baseline comparisons matched
- avoid adding machine-specific assumptions to public code

## Basic workflow

1. make the smallest correct change
2. run the relevant tests
3. update docs when system behavior changes
4. keep new public artifacts small and curated

## Validation

Use `uv` for Python commands.

```bash
PYTHONDONTWRITEBYTECODE=1 uv run python -m unittest discover -s tests_py -p 'test_*.py'
```
