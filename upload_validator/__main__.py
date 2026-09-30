"""Usage:  python -m upload_validator <file> [<file> ...] [--json]"""
import json
import sys
from .validator import validate_upload


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    as_json = "--json" in args
    files = [a for a in args if a != "--json"]
    if not files:
        print(__doc__)
        return 2
    worst = 0
    for f in files:
        r = validate_upload(f)
        if as_json:
            print(json.dumps({"file": f, **r.to_dict()}, indent=2))
        else:
            tag = "CASE 1 (VALID)  " if r.valid else "CASE 2 (INVALID)"
            print(f"[{tag}] {f}")
            print(f"    {r.message}")
            for d in r.details:
                print(f"    - {d}")
            print(f"    metrics: {r.metrics}")
        worst = max(worst, 0 if r.valid else 1)
    return worst


if __name__ == "__main__":
    sys.exit(main())
