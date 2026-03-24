"""
설정 파일 로딩 유틸리티

config.yaml 파일을 읽어서 Python 딕셔너리로 변환합니다.
"""

from pathlib import Path

import yaml


def load_config(path: str = "config.yaml") -> dict:
    """
    YAML 설정 파일을 로드합니다.

    Args:
        path: 설정 파일 경로 (기본값: config.yaml)

    Returns:
        설정 딕셔너리

    Example:
        >>> config = load_config("config.yaml")
        >>> seed = config["training"]["seed"]
        >>> print(seed)
        42
    """
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config
