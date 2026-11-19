"""Run a reproducible synthetic recurrent-gradient comparison."""

import argparse
import json

from .experiment import audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    try:
        print(json.dumps(audit(args.length, args.hidden, args.block_size, args.dropout, args.seed), indent=2, allow_nan=False))
    except (ValueError, ArithmeticError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
