"""Command-line user management for LiveGuard-EHMS.

Usage:
    python -m backend.auth.cli create-user <username>
    python -m backend.auth.cli list-users
    python -m backend.auth.cli delete-user <username>
    python -m backend.auth.cli change-password <username>
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
    subparsers.add_parser("list-users", help="List all user accounts")
    delete = subparsers.add_parser("delete-user", help="Delete a user account")
    delete.add_argument("username", help="Existing username to delete")
    change = subparsers.add_parser(
        "change-password", help="Change a user's password"
    )
    change.add_argument("username", help="Existing username whose password to change")
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


def list_users(args: argparse.Namespace) -> int:
    users = AuthService().list_users()
    if not users:
        print("No users found.")
        return 0
    fmt = "%-32s %-8s %-20s %s"
    print(fmt % ("USERNAME", "ROLE", "CREATED", "LAST LOGIN"))
    for user in users:
        print(
            fmt
            % (
                user["username"],
                user["role"],
                user["created_at"] or "-",
                user["last_login_at"] or "never",
            )
        )
    return 0


def delete_user(args: argparse.Namespace) -> int:
    ok, reason = AuthService().delete_user(args.username)
    if ok:
        print(f"Success: user '{args.username.strip().lower()}' deleted.")
        return 0
    print(f"Error: could not delete user: {reason}")
    return 1


def change_password(args: argparse.Namespace) -> int:
    # Verify the CURRENT password before prompting for a replacement, so a
    # wrong password (or missing account) fails fast without consuming the
    # rest of a piped stdin.
    current = _read_password("Current password: ")
    if current is None:
        print("Error: no password provided.")
        return 1
    auth = AuthService()
    ok, reason = auth.verify_password(args.username, current)
    if not ok:
        print(f"Error: could not change password: {reason}")
        return 1

    password = _read_password("New password: ")
    if password is None:
        print("Error: no password provided.")
        return 1
    confirm = _read_password("Confirm new password: ")
    if confirm is None:
        print("Error: no password provided.")
        return 1
    if password != confirm:
        print("Error: passwords do not match.")
        return 1

    ok, reason = auth.change_password(args.username, password)
    if ok:
        print(f"Success: password for '{args.username.strip().lower()}' changed.")
        return 0
    print(f"Error: could not change password: {reason}")
    return 1


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "create-user":
        return create_user(args)
    if args.command == "list-users":
        return list_users(args)
    if args.command == "delete-user":
        return delete_user(args)
    if args.command == "change-password":
        return change_password(args)
    return 1  # unreachable: a subcommand is required


if __name__ == "__main__":
    sys.exit(main())
