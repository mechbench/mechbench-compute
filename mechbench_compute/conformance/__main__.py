from __future__ import annotations

import argparse
import json
import sys

from mechbench_compute.conformance.report import check_extension, load_extension
from mechbench_compute.conformance.run_examples import read_inline, read_inputs_from


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mechbench_compute.conformance",
                                     description="Check an extension's declarations and run its examples twice.")
    parser.add_argument("extension", help="a package with a MANIFEST, `module:ATTR`, or an entry point name")
    parser.add_argument("--inputs", metavar="DIR", help="where the examples' referenced inputs are read from")
    parser.add_argument("--model", action="store_true", help="a model is available: run model operations too")
    args = parser.parse_args(argv)
    resolve = read_inputs_from(args.inputs) if args.inputs else read_inline
    result = check_extension(load_extension(args.extension), resolve_inputs=resolve, model=args.model)
    print(json.dumps(result.report(), indent=2))
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
