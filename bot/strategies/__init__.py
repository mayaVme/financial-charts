from ..config import StrategyConfig
from .base import Action, Signal, Strategy
from .ema_rsi import EmaRsiStrategy

STRATEGIES = {EmaRsiStrategy.name: EmaRsiStrategy}


def build_strategy(cfg: StrategyConfig) -> Strategy:
    try:
        return STRATEGIES[cfg.name](cfg)
    except KeyError:
        raise ValueError(f"Unknown strategy {cfg.name!r}; available: {sorted(STRATEGIES)}") from None


__all__ = ["Action", "Signal", "Strategy", "EmaRsiStrategy", "build_strategy"]
