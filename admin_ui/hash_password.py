from getpass import getpass

from admin_ui.auth import hash_password


if __name__ == "__main__":
    password = getpass("Admin password: ")
    confirmation = getpass("Confirm password: ")
    if password != confirmation:
        raise SystemExit("Passwords do not match")
    print(hash_password(password))
