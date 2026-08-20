"""Gated CLI for CC-NQE P4.7. Scientific runs are manual only."""
from __future__ import annotations

import argparse
import json

from cc_nqe_p4_7 import confirm, confirm_all, install_signal_handlers, preflight, run, screen, smoke, status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "smoke", "run", "screen", "status", "confirm", "confirm-all"))
    parser.add_argument("variant", nargs="?", choices=("C1", "C2", "C3"))
    parser.add_argument("--seed", type=int, choices=(2027, 2028))
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    install_signal_handlers()
    if args.command == "preflight": result = preflight()
    elif args.command == "smoke": result = smoke()
    elif args.command == "run":
        if not args.variant: parser.error("run requires C1, C2, or C3")
        result = run(args.variant)
    elif args.command == "screen": result = screen()
    elif args.command == "confirm":
        if args.seed is None: parser.error("confirm requires --seed 2027 or --seed 2028")
        if args.variant: parser.error("confirm does not accept a variant")
        result = confirm(args.seed)
    elif args.command == "confirm-all":
        if args.variant or args.seed is not None: parser.error("confirm-all takes no variant or seed")
        result = confirm_all()
    else: result = status()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
