"""CLI for the frozen P4.8 sealed-evaluation transaction."""
import argparse
import json

from cc_nqe_p4_8 import dry_run, preflight, prepare_artifacts, prepare_unlock, report, sealed_evaluate, status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("initialize", "preflight", "prepare-unlock", "sealed-evaluate", "resume-sealed-evaluate", "status", "report", "dry-run"))
    parser.add_argument("--unlock")
    parser.add_argument("--transaction")
    args = parser.parse_args()
    if args.command == "initialize": result = prepare_artifacts()
    elif args.command == "preflight": result = preflight()
    elif args.command == "prepare-unlock": result = prepare_unlock()
    elif args.command == "sealed-evaluate": result = sealed_evaluate(args.unlock)
    elif args.command == "resume-sealed-evaluate":
        if not args.transaction: parser.error("resume-sealed-evaluate requires --transaction")
        result = sealed_evaluate(None, args.transaction)
    elif args.command == "status": result = status()
    elif args.command == "report": result = report()
    else: result = dry_run()
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
