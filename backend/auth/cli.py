"""Command-line user management for LiveGuard-EHMS.

Usage:
    python -m backend.auth.cli create-user <username>
"""

import argparse
import getpass
import sys

from backend.auth.service import AuthService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m backend.auth.cli",
        description="LiveGuard user management",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create-user", help="Create a new user account")
    create.add_argument(
        "username",
        help="Username: 3-32 characters, lowercase letters, digits, underscore",
    )
    return parser


def _read_password(prompt: str) -> str | None:
    """Read one password line, honoring redirected stdin.

    On Windows, ``getpass.getpass`` reads keypresses from the console via
    ``msvcrt.getwch()`` and therefore blocks forever when stdin is a pipe or
    file (scripted provisioning). In that case read a line from stdin
    directly -- unlike ``fallback_getpass`` this neither warns about echo
    (nothing is echoed) nor touches the console. Returns ``None`` on
    EOF / Ctrl-C.
    """
    if sys.stdin is None or not sys.stdin.isatty():
        try:
            if prompt:
                sys.stderr.write(prompt)
                sys.stderr.flush()
            line = sys.stdin.readline() if sys.stdin is not None else ""
        except (EOFError, KeyboardInterrupt):
            return None
        if not line:
            return None
        return line.rstrip("\r\n")
    try:
        return getpass.getpass(prompt)
    except (EOFError, KeyboardInterrupt):
        return None


def create_user(args: argparse.Namespace) -> int:
    password = _read_password("Password: ")
    if password is None:
        print("Error: no password provided.")
        return 1
    confirm = _read_password("Confirm password: ")
    if confirm is None:
        print("Error: no password provided.")
        return 1
    if password != confirm:
        print("Error: passwords do not match.")
        return 1

    # provision() (not register()): the admin remedy must never be blocked by
    # the bootstrap-closed registration policy it points users to.
    ok, reason = AuthService().provision(args.username, password)
    if ok:
        print(f"Success: user '{args.username.strip().lower()}' created.")
        return 0
    print(f"Error: could not create user: {reason}")
    return 1


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "create-user":
        return create_user(args)
    return 1  # unreachable: a subcommand is required


if __name__ == "__main__":
    sys.exit(main())
