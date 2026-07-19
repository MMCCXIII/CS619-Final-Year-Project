import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATABASE_PATH
from src.data_access import seed_database


def main() -> None:
    seed_database(DATABASE_PATH)
    print(f"Database initialized: {DATABASE_PATH}")


if __name__ == "__main__":
    main()
