import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="config", config_name="train_linear")
def main(cfg: DictConfig) -> None:
    