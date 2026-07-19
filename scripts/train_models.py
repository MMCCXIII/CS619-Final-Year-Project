import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import METRICS_PATH, MODEL_BUNDLE_PATH
from src.modeling import train_and_save_models


def main() -> None:
    metrics = train_and_save_models()
    print(f"Model artifact saved: {MODEL_BUNDLE_PATH}")
    print(f"Metrics saved: {METRICS_PATH}")
    print(f"Best model: {metrics['best_model']}")


if __name__ == "__main__":
    main()
