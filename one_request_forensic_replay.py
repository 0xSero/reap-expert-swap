#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forensic_bundle import collect_bundle, write_bundle_and_visual


def main() -> int:
    ap = argparse.ArgumentParser(description="Collect and visualize one-request forensic artifacts.")
    ap.add_argument("--input-dir", required=True, help="Directory containing one-request run artifacts.")
    ap.add_argument("--output-dir", required=True, help="Directory for normalized forensic outputs.")
    ap.add_argument(
        "--forensic-json",
        default=None,
        help="Optional path to server-emitted forensic payload JSON.",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if expected forensic fields are missing.",
    )
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    forensic_json = Path(args.forensic_json) if args.forensic_json else None

    bundle = collect_bundle(input_dir=input_dir, forensic_json=forensic_json)
    json_path, md_path = write_bundle_and_visual(bundle=bundle, output_dir=output_dir)

    # Fail fast semantics for poisoned/engine-dead style runs.
    poisoned = bool(bundle.get("poison", {}).get("detected"))
    missing = bundle.get("expected_fields", {}).get("missing", [])
    crash_class = bundle.get("classification", {}).get("crash")

    print(
        json.dumps(
            {
                "forensic_bundle": str(json_path),
                "forensic_visual": str(md_path),
                "poisoned": poisoned,
                "crash_classification": crash_class,
                "missing_expected_fields": len(missing),
            },
            indent=2,
        )
    )

    if poisoned or crash_class in {"poisoned_runtime", "engine_dead"}:
        print("FAIL-FAST: poisoned or engine-dead classification detected.", file=sys.stderr)
        return 2
    if args.strict and missing:
        print(
            f"FAIL-FAST: strict mode with missing expected forensic fields: {missing}",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
