import os
import sys
import argparse
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("daily-phonk")

if __package__ in (None, ""):
    # 直接执行: python src/main.py
    from generator import DailyPhonkGenerator  # type: ignore
    from metadata import generate_publishing_assets
    from stats import log_generation
else:
    # 包方式执行: python -m src.main
    from .generator import DailyPhonkGenerator
    from .metadata import generate_publishing_assets
    from .stats import log_generation


def _default_config_path() -> str:
    here = Path(__file__).resolve().parent
    cfg = here / "config.yaml"
    # 允许从仓库根目录运行: python -m src.main
    if not cfg.exists():
        # 回退为绝对路径
        cfg = Path("/Users/wizout/op/daily-phonk/src/config.yaml")
    return str(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily Phonk Generator")
    parser.add_argument("--config", type=str, default=None, help="配置文件的绝对路径")
    parser.add_argument("--backend", type=str, choices=["neural", "dsp"], default=None, help="覆盖配置中的后端选择")
    parser.add_argument("--seed", type=int, default=None, help="随机数种子")
    args = parser.parse_args()

    config_path = args.config or os.environ.get("DAILY_PHONK_CONFIG") or _default_config_path()
    logger.info(f"使用配置: {config_path}")

    try:
        gen = DailyPhonkGenerator(config_path=config_path)
        
        # 覆盖配置中的 backend
        if args.backend:
            logger.info(f"覆盖配置的后端为: {args.backend}")
            gen.config.backend = args.backend

        # 执行生成
        result = gen.generate_once(seed=args.seed)
        
        # 加载完整配置字典供后续扩展使用
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)
            
        # 生成元数据与发布资产
        generate_publishing_assets(raw_config, result)
        
        # 记录统计/埋点
        log_generation(raw_config, result)
        
    except Exception as e:
        logger.exception("生成过程中发生未捕获的错误")
        sys.exit(1)


if __name__ == "__main__":
    # 允许直接 `python src/main.py` / `python -m src.main`
    sys.exit(main())


