"""
DSA 策略参数配置

Purpose: DSA 策略核心参数类，所有 pipeline 脚本从此导入

How to Run:
    python dsa_experiment/pipeline/dsa_config.py   # 验证

Side Effects: 无
"""

from dataclasses import dataclass


@dataclass
class DSAConfig:
    """DSA 策略参数配置

    从 features/dsa_bbmacd_24factors_viewer.py 迁移，
    作为 pipeline 模块的独立配置，解除对 features/ 的依赖。
    """
    prd: int = 50
    base_apt: float = 20.0
    use_adapt: bool = False
    vol_bias: float = 10.0
    atr_len: int = 50


if __name__ == "__main__":
    cfg = DSAConfig()
    print(f"DSAConfig: prd={cfg.prd}, base_apt={cfg.base_apt}, atr_len={cfg.atr_len}")
    print("✅ dsa_config 自测通过")
