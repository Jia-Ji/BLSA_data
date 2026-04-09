import hydra
from omegaconf import DictConfig
from contextlib import redirect_stdout
from pathlib import Path
import sys

from data.dataset_builder import ActiheartDatasetBuilder
from utils import _Tee

from data.dataset_builder import ActiheartDatasetBuilder
from models.reaction_models import LinearReactionModel
from models.outcome_models import LogisticOutcomeModel
from models.cox import CoxOutcomeModel
from models.two_step_pipeline import TwoStepPipeline


@hydra.main(version_base=None, config_path="config", config_name="train_linear")
def main(cfg: DictConfig) -> None:
    log_path = Path("main_log.txt")
    original_stdout = sys.stdout
    with open(log_path, "w", encoding="utf-8", newline="") as log_f:
        tee = _Tee(original_stdout, log_f)
        with redirect_stdout(tee):

                dataset_builder = ActiheartDatasetBuilder(**cfg.DatasetBuilder)

                reaction_model = LinearReactionModel(cfg.Model.reaction_model)

                outcome_model = CoxOutcomeModel(cfg.Model.cox_model)

                pipeline = TwoStepPipeline(
                    reaction_model=reaction_model,
                    outcome_model=outcome_model,
                    dataset_builder=dataset_builder,
                )

                pipeline.fit()

                # analysis_df = pipeline.get_analysis_dataset()
                # analysis_df = dataset_builder.add_age_group(analysis_df, age_col="age")

                # print(analysis_df.head())
                # print("Analysis dataset shape:", analysis_df.shape)

                # save models
                reaction_model.save(cfg.Save.reaction_model_path)
                outcome_model.save(cfg.Save.outcome_model_path)


if __name__ == "__main__":
    main()