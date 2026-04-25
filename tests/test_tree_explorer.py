"""
tests/test_tree_explorer.py

Argparse-based CLI for running tree_explorer queries interactively.

Usage:
    python -m tests.test_tree_explorer "passport fee"
    python -m tests.test_tree_explorer                            # interactive mode
"""

import argparse
import asyncio
import sys

from cogops.tools.graph.tree_explorer import tree_explorer_sync


def print_result(query: str, quiet: bool = False):
    try:
        result = tree_explorer_sync(query)
    except Exception as e:
        print(f"\nERROR: {e}\n")
        return

    if not quiet:
        print("\n" + "=" * 70)
        print(f'QUERY: "{query}"')
        print("=" * 70 + "\n")

    print(result)


def main():
    parser = argparse.ArgumentParser(
        description="Run tree_explorer queries and print Markdown results.",
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help='Search query (e.g. "passport fee"). If omitted, runs interactive mode.',
    )
    parser.add_argument(
        "--count",
        type=int,
        default=3,
        help="Number of queries to run interactively (default: 3).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only show the Markdown output, no headers or separators.",
    )
    args = parser.parse_args()

    if args.query:
        print_result(args.query, quiet=args.quiet)
    else:
        print("Tree Explorer CLI — interactive mode")
        print("Type a query and press Enter. Type 'quit' to exit.\n")
        for i in range(1, args.count + 1):
            query = input(f"[{i}/{args.count}] Query> ").strip()
            if not query or query.lower() in ("quit", "exit", "q"):
                break
            print_result(query, quiet=args.quiet)
            if i < args.count:
                print()


if __name__ == "__main__":
    main()
