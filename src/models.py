"""Model construction (declared in configs/scope.yaml; no silent defaults)."""
from __future__ import annotations

import yaml
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from proj_paths import CONFIG_DIR

SCOPE = yaml.safe_load((CONFIG_DIR / "scope.yaml").read_text(encoding="utf-8"))


def make_model(name: str, seed: int):
    """Instantiate a declared model with fixed seed; scaling inside a Pipeline where declared."""
    spec = SCOPE["models"][name]
    params = dict(spec.get("params", {}))
    params["random_state"] = seed if seed is not None else params.get("random_state")
    # some models use random_state / seed / random_seed naming
    if name in ("lightgbm",):
        params["random_state"] = seed
        params["verbose"] = -1
    if name in ("catboost",):
        params["random_seed"] = seed
        params.pop("random_state", None)
        params["verbose"] = 0
    if name == "xgboost":
        params["random_state"] = seed

    if name == "dummy_mean":
        return DummyRegressor(strategy="mean")
    if name == "ridge":
        return Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=params["alpha"]))])
    if name == "random_forest":
        return RandomForestRegressor(**params)
    if name == "xgboost":
        return XGBRegressor(**params)
    if name == "lightgbm":
        return LGBMRegressor(**params)
    if name == "catboost":
        return CatBoostRegressor(**params)
    raise ValueError(f"unknown model {name}")
