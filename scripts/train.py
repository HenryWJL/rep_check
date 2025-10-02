import hydra
from pathlib import Path
from omegaconf import OmegaConf


@hydra.main(
    config_path=str(Path(__file__).parent.parent.joinpath("rep_check", "configs")),
    version_base=None
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    cls = hydra.utils.get_class(cfg._target_)
    trainer = cls(cfg)
    trainer.run()


if __name__ == "__main__":
    main()