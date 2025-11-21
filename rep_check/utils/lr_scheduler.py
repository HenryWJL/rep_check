from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from typing import Optional


class CosineAnnealingLRWarmup(SequentialLR):

    def __init__(
        self,
        optimizer,
        warmup_epochs,
        T_max,
        start_factor: Optional[float] = 0.0,
        eta_min: Optional[float] = 0.0
    ) -> None:
        linear_scheduler = LinearLR(optimizer, start_factor, 1.0, warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max, eta_min)
        super().__init__(optimizer, [linear_scheduler, cosine_scheduler], [warmup_epochs])

    def step(self) -> None:
        return super().step()