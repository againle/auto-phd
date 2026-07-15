"""
通用工具函数

提供配置加载、日志设置、文件操作、时间戳等通用功能。
"""

import json
import os
import re
import logging
import logging.handlers
from pathlib import Path
from typing import Dict, Any, Optional, List

import yaml
from dotenv import load_dotenv


# ============================================================
# 配置加载
# ============================================================

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    加载 YAML 配置文件。

    Args:
        config_path: 配置文件路径，默认为项目根目录下的 config.yaml

    Returns:
        配置字典

    Raises:
        FileNotFoundError: 配置文件不存在时抛出
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(
            f"配置文件不存在: {config_file.absolute()}\n"
            f"请确保已创建 config.yaml 并填入正确的配置。"
        )

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"配置文件为空或格式错误: {config_file.absolute()}")

    return config


def load_env(env_path: str = ".env") -> None:
    """
    加载 .env 文件中的环境变量。

    使用 python-dotenv 将 .env 文件中的变量加载到 os.environ。
    如果 .env 文件不存在，会输出警告但不会中断程序。

    Args:
        env_path: .env 文件路径，默认为项目根目录下的 .env
    """
    env_file = Path(env_path)
    if env_file.exists():
        load_dotenv(dotenv_path=env_file, override=True)
    else:
        logging.warning(
            f".env 文件不存在: {env_file.absolute()}，"
            f"将使用系统环境变量。请参考 .env.example 创建。"
        )


# ============================================================
# 日志设置
# ============================================================

def setup_logging(log_config: Dict[str, Any]) -> None:
    """
    根据配置字典设置日志系统。

    支持日志级别、文件输出、按大小轮转等功能。

    Args:
        log_config: 日志配置字典，包含 level, file, max_size_mb, backup_count
    """
    level_name = log_config.get("level", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    log_file = log_config.get("file", "logs/agent.log")
    max_size_mb = log_config.get("max_size_mb", 10)
    backup_count = log_config.get("backup_count", 5)

    # 确保日志目录存在
    log_dir = Path(log_file).parent
    ensure_directory(str(log_dir))

    # 创建格式化器
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 创建按大小轮转的文件处理器
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_file,
        maxBytes=max_size_mb * 1024 * 1024,  # 转换为字节
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    # 避免重复添加处理器
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logging.info(f"日志系统已初始化，级别: {level_name}，文件: {log_file}")


# ============================================================
# 目录与文件操作
# ============================================================

def ensure_directory(path: str) -> Path:
    """
    确保目录存在，如果不存在则递归创建。

    Args:
        path: 目录路径

    Returns:
        Path 对象
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def save_json(data: Dict, filepath: str) -> None:
    """
    保存数据为格式化的 JSON 文件。

    自动确保父目录存在，使用 UTF-8 编码和缩进格式化。

    Args:
        data: 要保存的字典数据
        filepath: 目标文件路径
    """
    file_path = Path(filepath)
    ensure_directory(str(file_path.parent))

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    logging.debug(f"JSON 已保存: {filepath}")


def load_json(filepath: str) -> Dict:
    """
    从文件加载 JSON 数据。

    Args:
        filepath: JSON 文件路径

    Returns:
        解析后的字典

    Raises:
        FileNotFoundError: 文件不存在时抛出
    """
    file_path = Path(filepath)
    if not file_path.exists():
        raise FileNotFoundError(f"JSON 文件不存在: {filepath}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logging.debug(f"JSON 已加载: {filepath}")
    return data


# ============================================================
# 字符串工具
# ============================================================

def get_timestamp() -> str:
    """
    获取当前时间戳字符串。

    Returns:
        格式为 YYYYMMDD_HHMMSS 的时间戳，例如 "20260715_143022"
    """
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_filename(name: str) -> str:
    """
    将字符串转换为安全的文件名。

    去除或替换不能在文件名中使用的特殊字符：
    (反斜杠 / : * ? " < > |) 等。

    Args:
        name: 原始名称（如教授姓名）

    Returns:
        安全的文件名字符串，例如 "John_Doe"
    """
    # 替换路径分隔符和冒号
    name = name.replace("/", "_").replace("\\", "_").replace(":", "_")
    # 去除 Windows 文件名不允许的字符
    name = re.sub(r'[<>"|?*]', "", name)
    # 将连续空格和特殊符号替换为单个下划线
    name = re.sub(r'[\s]+', "_", name)
    # 去除首尾的下划线和点
    name = name.strip("_ .")
    # 如果处理后为空，返回默认名称
    if not name:
        name = "unknown"
    return name


# ============================================================
# 模型路由工具
# ============================================================

def get_model_for_task(task_type: str, config: Dict[str, Any]) -> str:
    """
    根据任务类型返回推荐的模型名称。

    Args:
        task_type: 任务类型，如 "search_discussion", "paper_summary", "email_generation"
        config: 加载的完整配置字典

    Returns:
        模型名称，如 "deepseek" 或 "gpt_mini_backup"

    Raises:
        ValueError: 任务类型未知或映射不存在时抛出
    """
    llm_config = config.get("llm", {})
    task_mapping = llm_config.get("task_model_mapping", {})

    if task_type in task_mapping:
        model = task_mapping[task_type]
        logger = logging.getLogger(__name__)
        logger.debug(f"任务 '{task_type}' → 模型 '{model}'")
        return model

    # 回退到默认模型
    default = llm_config.get("default_provider", "deepseek")
    logging.getLogger(__name__).warning(
        f"任务 '{task_type}' 无专用映射，使用默认模型 '{default}'"
    )
    return default


def validate_api_key(provider: str, api_key: Optional[str] = None) -> bool:
    """
    验证 API 密钥是否有效。

    通过发送一个最小的测试请求来验证密钥。

    Args:
        provider: 提供商名称 ("deepseek" 或 "openai")
        api_key: API 密钥，若为 None 则从环境变量读取

    Returns:
        True 表示有效，False 表示无效
    """
    import requests

    if api_key is None:
        env_key_map = {
            "deepseek": "DEEPSEEK_API_KEY",
            "openai": "OPENAI_API_KEY",
        }
        env_var = env_key_map.get(provider.lower(), "")
        api_key = os.getenv(env_var, "")

    if not api_key or "YOUR_" in api_key:
        logging.getLogger(__name__).warning(
            f"提供商 '{provider}' 的 API key 未设置或为占位符"
        )
        return False

    # 各提供商的验证端点
    endpoints = {
        "deepseek": "https://api.deepseek.com/v1/models",
        "openai": "https://api.openai.com/v1/models",
    }

    endpoint = endpoints.get(provider.lower())
    if not endpoint:
        logging.getLogger(__name__).warning(f"未知的提供商: {provider}")
        return False

    try:
        response = requests.get(
            endpoint,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        if response.status_code == 200:
            logging.getLogger(__name__).info(f"✅ {provider} API key 验证成功")
            return True
        elif response.status_code == 401:
            logging.getLogger(__name__).warning(f"❌ {provider} API key 无效 (401)")
            return False
        else:
            logging.getLogger(__name__).warning(
                f"⚠️ {provider} API key 验证返回 {response.status_code}"
            )
            return response.status_code < 500  # 非服务器错误视为 key 可能有效
    except Exception as e:
        logging.getLogger(__name__).error(f"❌ {provider} API key 验证失败: {e}")
        return False


def format_cost(cost: float) -> str:
    """
    格式化成本显示。

    Args:
        cost: 美元金额，如 0.0034

    Returns:
        格式化字符串，如 "$0.0034"
    """
    if cost < 0.01:
        return f"${cost:.6f}"
    elif cost < 1.0:
        return f"${cost:.4f}"
    else:
        return f"${cost:.2f}"


def get_available_models(config: Dict[str, Any]) -> List[str]:
    """
    返回配置中所有可用模型的名称列表。

    Args:
        config: 加载的完整配置字典

    Returns:
        模型名称列表，如 ["deepseek", "gpt_mini_backup"]
    """
    llm_config = config.get("llm", {})
    models = llm_config.get("models", {})
    return list(models.keys())


# ============================================================
# 程序入口测试（直接运行此文件时执行）
# ============================================================

if __name__ == "__main__":
    # 简单测试
    print(f"时间戳: {get_timestamp()}")
    print(f"安全文件名测试 'Dr. Jane Smith / AI Lab': {safe_filename('Dr. Jane Smith / AI Lab')}")

    # 测试配置加载
    try:
        config = load_config()
        print(f"LLM 默认模型: {config['llm']['default_provider']}")
        print(f"可用模型: {get_available_models(config)}")
        print(f"search_discussion → {get_model_for_task('search_discussion', config)}")
        print(f"email_generation → {get_model_for_task('email_generation', config)}")
        print(f"unknown_task → {get_model_for_task('unknown_task', config)}")
        print(f"成本格式化: {format_cost(0.00034)} / {format_cost(0.12)} / {format_cost(2.50)}")
        print("配置加载成功！")
    except FileNotFoundError:
        print("config.yaml 不存在（这是正常的，如果尚未创建）")

    # 测试环境变量加载
    load_env()
    print(f"OPENAI_API_KEY 是否设置: {'是' if os.getenv('OPENAI_API_KEY') else '否（正常，如果尚未配置）'}")
