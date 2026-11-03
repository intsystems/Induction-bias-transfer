"""Plot the hidden state initialization and prior experiment."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_pattern_transfer"))

from plotting import main


if __name__ == "__main__":
    main("results/pattern_hidden_prior/metrics.json")
