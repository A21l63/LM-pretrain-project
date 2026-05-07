"""Run the required experiment set."""

from src.experiments import REQUIRED_EXPERIMENTS, run_many


if __name__ == "__main__":
    run_many(REQUIRED_EXPERIMENTS)
