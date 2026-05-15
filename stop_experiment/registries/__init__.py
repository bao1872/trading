# -*- coding: utf-8 -*-
"""
注册表加载模块 — 统一的 registry/profile 读写入口

Purpose:
    提供 feature_registry / model_registry / strategy_registry / profiles
    的加载、解析和查询接口。所有模块通过此文件访问版本信息，
    禁止直接读 JSON 文件。

Usage:
    from stop_experiment.registries import (
        load_profile, resolve_profile_params, resolve_model_paths,
        load_feature_registry, load_model_registry, load_strategy_registry,
    )

Side Effects:
    - 首次加载时有文件 I/O，后续调用使用模块级缓存
"""

from __future__ import annotations

import json
import os
from copy import deepcopy

_REGISTRY_DIR = os.path.dirname(os.path.abspath(__file__))

_cache: dict = {}


def _load_json(filename: str) -> dict:
    if filename not in _cache:
        path = os.path.join(_REGISTRY_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"注册表文件不存在: {path}")
        with open(path, "r", encoding="utf-8") as f:
            _cache[filename] = json.load(f)
    return deepcopy(_cache[filename])


def load_feature_registry() -> dict:
    return _load_json("feature_registry.json")


def load_model_registry() -> dict:
    return _load_json("model_registry.json")


def load_strategy_registry() -> dict:
    return _load_json("strategy_registry.json")


def load_profiles() -> dict:
    return _load_json("profiles.json")


def load_profile(profile_name: str) -> dict:
    """加载指定 profile 并解析其引用的 feature/model/strategy"""
    profiles = load_profiles()
    if profile_name not in profiles.get("profiles", {}):
        raise KeyError(f"Profile '{profile_name}' 不存在，可用: {list(profiles.get('profiles', {}).keys())}")

    profile = dict(profiles["profiles"][profile_name])

    feature_reg = load_feature_registry()
    fv = profile.get("feature_version")
    if fv and fv in feature_reg.get("versions", {}):
        profile["feature_config"] = feature_reg["versions"][fv]

    model_reg = load_model_registry()
    mv = profile.get("model_version")
    if mv and mv in model_reg.get("models", {}):
        profile["model_config"] = model_reg["models"][mv]

    strategy_reg = load_strategy_registry()
    sv = profile.get("strategy_version")
    if sv and sv in strategy_reg.get("strategies", {}):
        profile["strategy_config"] = strategy_reg["strategies"][sv]

    return profile


def resolve_profile_params(profile_name: str) -> dict:
    """从 profile → strategy_registry → 返回 strategy params dict"""
    strategy_reg = load_strategy_registry()
    profile = load_profile(profile_name)
    sv = profile.get("strategy_version")
    if sv and sv in strategy_reg.get("strategies", {}):
        params = dict(strategy_reg["strategies"][sv].get("params", {}))
        params["profile"] = profile_name
        params["strategy_version"] = sv
        return params
    return {}


def resolve_model_paths(profile_name: str) -> dict:
    """从 profile → model_registry → 返回模型文件路径 dict"""
    model_reg = load_model_registry()
    profile = load_profile(profile_name)
    mv = profile.get("model_version")
    if mv and mv in model_reg.get("models", {}):
        return dict(model_reg["models"][mv].get("artifacts", {}))
    return {}


def resolve_prediction_store_path(profile_name: str, date: str) -> str:
    """解析 prediction_store 路径: prediction_store/{profile}/{fv}/{mv}/{date}.parquet"""
    import os as _os
    profile = load_profile(profile_name)
    fv = profile.get("feature_version", "unknown")
    mv = profile.get("model_version", "unknown")

    stop_root = _os.path.dirname(_REGISTRY_DIR)
    base = _os.path.join(stop_root, "output", "prediction_store", profile_name, fv, mv)
    return _os.path.join(base, f"{date}.parquet")


def resolve_manifest_path(profile_name: str) -> str:
    """解析 prediction_store manifest.json 路径"""
    import os as _os
    profile = load_profile(profile_name)
    fv = profile.get("feature_version", "unknown")
    mv = profile.get("model_version", "unknown")

    stop_root = _os.path.dirname(_REGISTRY_DIR)
    base = _os.path.join(stop_root, "output", "prediction_store", profile_name, fv, mv)
    return _os.path.join(base, "manifest.json")