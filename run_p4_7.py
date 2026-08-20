"""Gated CLI for CC-NQE P4.7. Scientific runs are manual only."""
from __future__ import annotations

import argparse
import json

from cc_nqe_p4_7 import install_signal_handlers, preflight, run, screen, smoke, status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "smoke", "run", "screen", "status"))
    parser.add_argument("variant", nargs="?", choices=("C1", "C2", "C3"))
    args = parser.parse_args()
    install_signal_handlers()
    if args.command == "preflight": result = preflight()
    elif args.command == "smoke": result = smoke()
    elif args.command == "run":
        if not args.variant: parser.error("run requires C1, C2, or C3")
        result = run(args.variant)
    elif args.command == "screen": result = screen()
    else: result = status()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
