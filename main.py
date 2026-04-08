import hydra
from omegaconf import DictConfig
from contextlib import redirect_stdout
from pathlib import Path
import sys

from data.create_dataset import ActiheartDatasetBuilder
from utils import _Tee


@hydra.main(version_base=None, config_path="config", config_name="train_linear")
def main(cfg: DictConfig) -> None:
    log_path = Path("main_log.txt")
    original_stdout = sys.stdout
    with open(log_path, "w", encoding="utf-8", newline="") as log_f:
        tee = _Tee(original_stdout, log_f)
        with redirect_stdout(tee):
            print("Starting main...")

            # Initilize dataset builder
            dataset = ActiheartDatasetBuilder(**cfg.data)
            print("Dataset initialized.")

            print("Preprocessing minute data...")
            # preprocess minute-level heart rate and pa data
            preprocessed_minute_data = dataset.preprocess_minute_data()
            print("Preprocessing done.")

            print("Building subject features...")
            # Create subject-level features
            subject_features = dataset.build_subject_features()
            print("Subject features done.")


if __name__ == "__main__":
    main()