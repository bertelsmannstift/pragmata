import argparse
import sys

import argilla as rg


def delete_argilla_user(username):
    try:
        client = rg.Argilla(api_url="http://localhost:6900", api_key="argilla.apikey")
        user = client.users(username)
        if user is None:
            print(f"User '{username}' not found.")
            sys.exit(1)
        user.delete()
        print(f"Successfully deleted user: {username}")

    except Exception as e:
        print(f"Error: Could not delete user '{username}'.")
        print(f"Details: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete an Argilla user by username.")
    parser.add_argument("username", help="Username to delete")
    args = parser.parse_args()

    delete_argilla_user(args.username)
