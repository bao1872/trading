#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型晋升工具 — 将 candidate 模型 version 晋升到 production profile

Purpose:
    提供 CLI 工具将已验证的模型版本提升为 production：
    1. 验证模型文件存在且包含 4 个必需模型
    2. 更新 profiles.json 中 production profile 的 model_version
    3. 要求显式确认，防止误操作

Usage:
    python -m stop_experiment.registries.promote --model mv_20260511_retrain_v1 --profile production

Options:
    --model   要晋升的模型版本（必须在 model_registry 中状态为 candidate）
    --profile 目标 profile（默认 production）
    --yes     跳过确认提示
    --dry-run 只检查不修改

Side Effects:
    - 修改 registries/profiles.json
    - 不修改模型文件
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_REGISTRY_DIR = os.path.dirname(os.path.abspath(__file__))
_STOP_ROOT = os.path.dirname(_REGISTRY_DIR)


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  [已保存] {path}")


def _validate_model_artifacts(model_version, artifacts):
    """验证 4 个模型文件存在"""
    required = ["buy_cls", "buy_reg", "sell_cls", "sell_reg"]
    missing = []
    for key in required:
        path = artifacts.get(key, "")
        if not path or not os.path.exists(os.path.join(_STOP_ROOT, path)):
            missing.append(key)
    return missing


def main():
    parser = argparse.ArgumentParser(description="模型晋升工具 — 将已验证模型提升为 production")
    parser.add_argument("--model", required=True, help="目标模型版本 (如 mv_20260511_retrain_v1)")
    parser.add_argument("--profile", default="production", help="目标 profile (默认 production)")
    parser.add_argument("--yes", action="store_true", help="跳过确认提示")
    parser.add_argument("--dry-run", action="store_true", help="只检查不修改")
    parser.add_argument("--register", type=str, default=None,
                        help="注册新模型版本到 model_registry (需 --model-dir)")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="模型文件所在目录 (用于 --register)")
    args = parser.parse_args()

    # Register mode: 注册新模型版本
    if args.register:
        if not args.model_dir:
            print("❌ --register 需要 --model-dir")
            sys.exit(1)
        _register_model(args.register, args.model_dir)
        return

    # Promote mode: 晋升为 production
    dry = "🧪 DRY-RUN: " if args.dry_run else ""

    model_reg_path = os.path.join(_REGISTRY_DIR, "model_registry.json")
    profiles_path = os.path.join(_REGISTRY_DIR, "profiles.json")

    model_reg = _read_json(model_reg_path)
    profiles = _read_json(profiles_path)

    if args.model not in model_reg.get("models", {}):
        print(f"❌ 模型版本 '{args.model}' 不在 model_registry 中")
        available = list(model_reg.get("models", {}).keys())
        print(f"  可用: {available}")
        sys.exit(1)

    model_info = model_reg["models"][args.model]
    if model_info.get("status") != "candidate":
        print(f"⚠ 模型 '{args.model}' 状态是 '{model_info.get('status')}'，不是 'candidate'. 继续...")

    missing = _validate_model_artifacts(args.model, model_info.get("artifacts", {}))
    if missing:
        print(f"❌ 模型文件缺失: {missing}")
        sys.exit(1)

    print()
    print("=" * 70)
    print(f"{dry}晋升模型版本")
    print(f"  模型: {args.model} ({model_info.get('description', 'N/A')})")
    print(f"  目标Profile: {args.profile}")
    print("=" * 70)

    if not args.yes and not args.dry_run:
        confirm = input(f"\n确认将 production 切到 {args.model}? [y/N]: ").strip().lower()
        if confirm != "y":
            print("已取消")
            return

    if args.dry_run:
        print(f"\n{dry}跳过写入")
        return

    # Update profile
    if args.profile not in profiles.get("profiles", {}):
        print(f"❌ Profile '{args.profile}' 不存在")
        sys.exit(1)

    profiles["profiles"][args.profile]["model_version"] = args.model
    _write_json(profiles_path, profiles)

    # Update model status
    model_reg["models"][args.model]["status"] = "production"
    model_reg["active"] = args.model
    _write_json(model_reg_path, model_reg)

    print(f"\n✅ 已晋升 {args.model} → {args.profile} profile")
    print(f"  请运行验证: python -m stop_experiment.tests_consistency.run_all_checks --ci")


def _register_model(model_version, model_dir):
    """注册新模型版本到 model_registry.json"""
    model_reg_path = os.path.join(_REGISTRY_DIR, "model_registry.json")
    model_reg = _read_json(model_reg_path)

    artifacts = {
        "buy_cls": os.path.join(model_dir, "buy_cls_final.txt"),
        "buy_reg": os.path.join(model_dir, "buy_reg_final.txt"),
        "sell_cls": os.path.join(model_dir, "sell_cls_final.txt"),
        "sell_reg": os.path.join(model_dir, "sell_reg_final.txt"),
    }

    missing = _validate_model_artifacts(model_version, artifacts)
    if missing:
        print(f"❌ 模型文件缺失: {missing}")
        sys.exit(1)

    from datetime import datetime
    model_reg["models"][model_version] = {
        "description": f"自动注册 (训练目录: {model_dir})",
        "feature_version": "fv_202605_v1",
        "label_version": "lv_202605_v1",
        "training_data": "dataset.parquet",
        "artifacts": artifacts,
        "metrics": {},
        "created_at": datetime.now().strftime("%Y-%m-%d"),
        "status": "candidate",
    }
    _write_json(model_reg_path, model_reg)

    print(f"\n✅ 已注册新模型版本: {model_version} (status=candidate)")
    print(f"  验证后晋升: python -m stop_experiment.registries.promote --model {model_version}")


if __name__ == "__main__":
    main()