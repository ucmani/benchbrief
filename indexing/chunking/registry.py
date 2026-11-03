from .base import ChunkingStrategy
from .heuristic_strategy import HeuristicStrategy
from .fixed_size_strategy import FixedSizeStrategy
from .llm_strategy import LLMStrategy

_STRATEGY_CLASSES: dict[str, type[ChunkingStrategy]] = {
    "heuristic": HeuristicStrategy,
    "fixed_size": FixedSizeStrategy,
    "llm": LLMStrategy,
}

def get_strategy(name: str, **kwargs) -> ChunkingStrategy:
    if name not in _STRATEGY_CLASSES:
        available_strategies = ", ".join(_STRATEGY_CLASSES.keys())
        raise ValueError(f"Unknown strategy '{name}'. Available strategies: {available_strategies}")

    strategy_class = _STRATEGY_CLASSES[name]
    return strategy_class(**kwargs)

def get_all_strategies(names: list[str] | None = None, **kwargs_by_name) -> dict[str, ChunkingStrategy]:
    selected_names = names if names is not None else list(_STRATEGY_CLASSES.keys())
    return {
        name: get_strategy(name, **kwargs_by_name.get(name, {}))
        for name in selected_names
    }