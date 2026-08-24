from __future__ import annotations

import argparse
import json

from cc_nqe_ab_confirmatory import A_ARMS, B_VARIANTS, aggregate, prepare, run_cell


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    run = sub.add_parser("run")
    run.add_argument("kind", choices=("A", "B"))
    run.add_argument("name", choices=A_ARMS + B_VARIANTS)
    run.add_argument("seed", type=int, choices=(2027, 2028))
    sub.add_parser("aggregate")
    args = parser.parse_args()
    result = prepare() if args.command == "prepare" else aggregate() if args.command == "aggregate" else run_cell(args.kind, args.name, args.seed)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
