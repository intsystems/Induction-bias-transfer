"""Plot the learned augmentation experiment from its JSON results."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_pattern_transfer"))

from plotting import main


if __name__ == "__main__":
    main("results/pattern_augmentation/metrics.json")
