# Public staging validation report

## Safety scan

A forbidden-pattern scan was run against the staged public tree for private-machine references, private network references, private hostnames, and private activation-corpus residue.

Result:

- **no forbidden matches in staged source files**

## Functional validation

```bash
PYTHONDONTWRITEBYTECODE=1 uv run python -m unittest \
  tests_py.test_dynamic_reap \
  tests_py.test_multiturn_evaluator \
  tests_py.test_router_activity \
  tests_py.test_profiled_floor_plan \
  tests_py.test_support_router

PYTHONDONTWRITEBYTECODE=1 uv run python -m py_compile scripts/*.py tests_py/*.py
```

Results:

- unittest: **45 tests passed**
- py_compile: **passed**
