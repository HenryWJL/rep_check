import hydra
from pathlib import Path
from omegaconf import OmegaConf
from rep_check.models.gcn import STGCN

OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    config_path=str(Path(__file__).parent.parent.joinpath("rep_check", "configs")),
    version_base=None
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    model: STGCN = hydra.utils.instantiate(cfg.model)

    import torch
    x = torch.rand(2, 4, 30, 23)
    print(model(x).shape)


if __name__ == "__main__":
    main()