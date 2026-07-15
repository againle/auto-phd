"""
多模型 LLM 客户端（支持自动故障切换）

功能：
- 支持 DeepSeek / OpenAI 等多提供商
- 主模型失败时自动切换到备用模型
- 完整的成本追踪和调用日志
- 基于 tenacity 的智能重试
- 支持流式输出
"""

import json
import os
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass, field

import yaml
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
from pydantic import BaseModel, Field, ValidationError

# 兼容直接运行 (python scripts/llm_client.py) 和模块运行 (python -m scripts.llm_client)
try:
    from scripts.utils import load_config, ensure_directory
except (ModuleNotFoundError, ImportError):
    import sys
    _parent = Path(__file__).resolve().parent.parent
    if str(_parent) not in sys.path:
        sys.path.insert(0, str(_parent))
    from scripts.utils import load_config, ensure_directory

logger = logging.getLogger(__name__)


# ============================================================
# 枚举与数据类
# ============================================================

class TaskType(Enum):
    """LLM 任务类型枚举"""
    SEARCH_DISCUSSION = "search_discussion"
    PAPER_SUMMARY = "paper_summary"
    EMAIL_GENERATION = "email_generation"
    GENERAL = "general"


@dataclass
class LLMCallRecord:
    """记录单次 LLM 调用"""
    task_type: str
    model_used: str
    provider: str
    input_tokens: int
    output_tokens: int
    cost: float
    success: bool
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    latency_seconds: float = 0.0
    error_message: str = ""


# ============================================================
# Pydantic 配置模型
# ============================================================

class ModelPricing(BaseModel):
    """模型定价"""
    input_per_1k: float = 0.0
    output_per_1k: float = 0.0


class ModelConfig(BaseModel):
    """单个模型配置"""
    provider: str
    api_key: str
    base_url: str
    model: str
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: int = 60
    pricing: ModelPricing = Field(default_factory=ModelPricing)


class LLMConfig(BaseModel):
    """LLM 总体配置"""
    default_provider: str = "deepseek"
    failover_threshold: int = 2
    global_max_retries: int = 3
    models: Dict[str, ModelConfig] = Field(default_factory=dict)
    task_model_mapping: Dict[str, str] = Field(default_factory=dict)


# ============================================================
# 成本追踪器
# ============================================================

class CostTracker:
    """
    LLM 调用成本追踪器。

    按模型累计 token 使用量和费用，支持查询总成本。
    """

    def __init__(self):
        self._records: List[LLMCallRecord] = []
        self._model_costs: Dict[str, float] = {}       # {model_name: total_cost}
        self._model_tokens: Dict[str, Dict[str, int]] = {}  # {model_name: {input: N, output: N}}

    def add_record(self, record: LLMCallRecord) -> None:
        """添加一条调用记录"""
        self._records.append(record)

        model = record.model_used
        if model not in self._model_costs:
            self._model_costs[model] = 0.0
            self._model_tokens[model] = {"input": 0, "output": 0}

        self._model_costs[model] += record.cost
        self._model_tokens[model]["input"] += record.input_tokens
        self._model_tokens[model]["output"] += record.output_tokens

    def get_total_cost(self) -> float:
        """返回累计总成本（美元）"""
        return sum(self._model_costs.values())

    def get_model_cost(self, model_name: str) -> float:
        """返回指定模型的成本"""
        return self._model_costs.get(model_name, 0.0)

    def get_model_tokens(self, model_name: str) -> Dict[str, int]:
        """返回指定模型的 token 统计"""
        return self._model_tokens.get(model_name, {"input": 0, "output": 0})

    def get_summary(self) -> Dict[str, Any]:
        """返回完整成本汇总"""
        return {
            "total_cost_usd": round(self.get_total_cost(), 6),
            "total_calls": len(self._records),
            "successful_calls": sum(1 for r in self._records if r.success),
            "failed_calls": sum(1 for r in self._records if not r.success),
            "per_model": {
                model: {
                    "cost_usd": round(cost, 6),
                    "input_tokens": self._model_tokens.get(model, {}).get("input", 0),
                    "output_tokens": self._model_tokens.get(model, {}).get("output", 0),
                }
                for model, cost in self._model_costs.items()
            },
        }

    def reset(self) -> None:
        """重置所有统计"""
        self._records.clear()
        self._model_costs.clear()
        self._model_tokens.clear()

    @property
    def records(self) -> List[LLMCallRecord]:
        return self._records


# ============================================================
# LLM 客户端
# ============================================================

class LLMClient:
    """
    多模型 LLM 客户端，支持自动故障切换。

    使用示例:
        client = LLMClient()
        response = client.call(
            messages=[{"role": "user", "content": "Hello"}],
            task_type=TaskType.GENERAL,
        )
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化 LLM 客户端。

        Args:
            config_path: 配置文件路径
        """
        # 加载并解析配置
        raw_config = load_config(config_path)
        self._raw_config = raw_config

        try:
            self.config = LLMConfig(**raw_config.get("llm", {}))
        except ValidationError as e:
            logger.error(f"LLM 配置验证失败: {e}")
            raise ValueError(f"LLM 配置格式错误，请检查 config.yaml: {e}") from e

        # 初始化
        self.cost_tracker = CostTracker()
        self._clients: Dict[str, OpenAI] = {}      # 缓存已创建的 OpenAI 客户端
        self._fail_count: Dict[str, int] = {}       # {model_name: consecutive_failures}
        self.last_model_used: str = ""              # 上次使用的模型名

        # 初始化所有模型的客户端
        self._init_clients()

        # 调用日志文件
        log_dir = ensure_directory("logs")
        self._call_log_file = log_dir / "llm_calls.log"

        logger.info(
            f"LLM 客户端已初始化，默认模型: {self.config.default_provider}，"
            f"可用模型: {list(self.config.models.keys())}，"
            f"故障切换阈值: {self.config.failover_threshold}"
        )

    # --------------------------------------------------------
    # 初始化
    # --------------------------------------------------------

    def _init_clients(self) -> None:
        """为每个配置的模型创建 OpenAI 客户端"""
        for model_name, model_conf in self.config.models.items():
            # 允许从环境变量覆盖 API key
            env_key_map = {
                "deepseek": "DEEPSEEK_API_KEY",
                "openai": "OPENAI_API_KEY",
            }
            env_var = env_key_map.get(model_conf.provider, "")
            api_key = os.getenv(env_var) or model_conf.api_key

            if "YOUR_" in api_key:
                logger.warning(
                    f"模型 '{model_name}' 的 API key 仍为占位符，"
                    f"请在 config.yaml 或环境变量 {env_var} 中设置真实密钥"
                )

            self._clients[model_name] = OpenAI(
                api_key=api_key,
                base_url=model_conf.base_url,
                timeout=model_conf.timeout,
            )
            self._fail_count[model_name] = 0

    # --------------------------------------------------------
    # 主调用接口
    # --------------------------------------------------------

    def call(
        self,
        messages: List[Dict[str, str]],
        task_type: TaskType = TaskType.GENERAL,
        max_retries: int = 3,
        temperature: Optional[float] = None,
        stream: bool = False,
    ) -> str:
        """
        调用 LLM 主入口，自动处理模型选择和故障切换。

        Args:
            messages: OpenAI 格式的消息列表
            task_type: 任务类型，用于选择模型
            max_retries: 最大重试次数（覆盖全局配置）
            temperature: 温度参数，None 则使用模型默认值
            stream: 是否流式输出

        Returns:
            模型生成的文本

        Raises:
            RuntimeError: 所有模型均不可用时抛出
        """
        task_str = task_type.value

        # 根据任务类型选择模型
        primary_model = self._get_model_for_task(task_str)
        logger.info(f"[{task_str}] 主模型: {primary_model}")

        # 构建尝试队列：主模型 → 备用模型
        model_queue = self._build_model_queue(primary_model)

        last_error = None

        for attempt_idx, model_name in enumerate(model_queue):
            model_conf = self.config.models[model_name]

            # 检查该模型是否已达故障切换阈值
            fail_count = self._fail_count.get(model_name, 0)
            if fail_count >= self.config.failover_threshold:
                logger.warning(
                    f"模型 '{model_name}' 已连续失败 {fail_count} 次，跳过"
                )
                continue

            temp = temperature if temperature is not None else model_conf.temperature

            try:
                print(f"[{model_conf.provider.upper()}] 正在生成... (模型: {model_conf.model})")

                content, usage = self._call_with_retry(
                    model_name=model_name,
                    model_conf=model_conf,
                    messages=messages,
                    temperature=temp,
                    max_retries=max_retries,
                    stream=stream,
                )

                # 成功：重置失败计数，记录日志
                self._fail_count[model_name] = 0
                self.last_model_used = model_name

                record = self._build_record(
                    task_type=task_str,
                    model_name=model_name,
                    model_conf=model_conf,
                    usage=usage,
                    success=True,
                )
                self.cost_tracker.add_record(record)
                self._log_call(record)

                logger.info(
                    f"[{task_str}] 成功 | 模型: {model_name} | "
                    f"tokens: {usage.get('total_tokens', '?')} | "
                    f"成本: ${record.cost:.6f}"
                )

                return content

            except Exception as e:
                self._fail_count[model_name] = self._fail_count.get(model_name, 0) + 1
                last_error = e

                record = self._build_record(
                    task_type=task_str,
                    model_name=model_name,
                    model_conf=model_conf,
                    usage={},
                    success=False,
                    error=str(e),
                )
                self.cost_tracker.add_record(record)
                self._log_call(record)

                logger.warning(
                    f"[{task_str}] 模型 '{model_name}' 调用失败 "
                    f"(连续失败 {self._fail_count[model_name]} 次): {e}"
                )

                if attempt_idx < len(model_queue) - 1:
                    logger.info(f"正在切换到下一个模型: {model_queue[attempt_idx + 1]}")
                    # 短暂等待后再切换
                    time.sleep(1.0)

        # 所有模型都失败
        raise RuntimeError(
            f"所有模型均调用失败。"
            f"任务: {task_str}，"
            f"尝试模型: {model_queue}，"
            f"最后错误: {last_error}"
        )

    # --------------------------------------------------------
    # 模型选择
    # --------------------------------------------------------

    def _get_model_for_task(self, task_type: str) -> str:
        """根据任务类型获取对应模型名"""
        mapping = self.config.task_model_mapping
        if task_type in mapping:
            return mapping[task_type]
        return self.config.default_provider

    def _build_model_queue(self, primary_model: str) -> List[str]:
        """构建模型尝试队列：主模型 → 其余模型（按配置顺序）"""
        queue = [primary_model]
        for model_name in self.config.models:
            if model_name not in queue:
                queue.append(model_name)
        return queue

    # --------------------------------------------------------
    # 实际调用（带重试）
    # --------------------------------------------------------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(Exception),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _call_single_model(
        self,
        model_name: str,
        model_conf: ModelConfig,
        messages: List[Dict[str, str]],
        temperature: float,
        stream: bool = False,
    ) -> Tuple[str, Dict]:
        """
        调用单个模型（被 _call_with_retry 包装）。

        Args:
            model_name: 配置中的模型名
            model_conf: 模型配置
            messages: 消息列表
            temperature: 温度
            stream: 是否流式

        Returns:
            (响应文本, usage 字典)

        Raises:
            Exception: 各类 API 错误
        """
        client = self._clients[model_name]

        if stream:
            # 流式输出
            response = client.chat.completions.create(
                model=model_conf.model,
                messages=messages,
                temperature=temperature,
                max_tokens=model_conf.max_tokens,
                stream=True,
            )

            full_text = ""
            usage_info = {}
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    full_text += text
                    print(text, end="", flush=True)
                # 最后一个 chunk 可能包含 usage
                if hasattr(chunk, "usage") and chunk.usage:
                    usage_info = {
                        "prompt_tokens": chunk.usage.prompt_tokens or 0,
                        "completion_tokens": chunk.usage.completion_tokens or 0,
                        "total_tokens": chunk.usage.total_tokens or 0,
                    }
            print()  # 换行
        else:
            # 非流式输出
            response = client.chat.completions.create(
                model=model_conf.model,
                messages=messages,
                temperature=temperature,
                max_tokens=model_conf.max_tokens,
                stream=False,
            )

            full_text = response.choices[0].message.content or ""
            usage_info = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            }

        return full_text, usage_info

    def _call_with_retry(
        self,
        model_name: str,
        model_conf: ModelConfig,
        messages: List[Dict[str, str]],
        temperature: float,
        max_retries: int,
        stream: bool,
    ) -> Tuple[str, Dict]:
        """
        带重试的单模型调用包装器。

        使用 tenacity 的 retry 装饰器处理临时性错误（429限流等），
        对于不可重试的错误（401认证失败等）直接抛出。
        """
        start_time = time.time()

        try:
            content, usage = self._call_single_model(
                model_name=model_name,
                model_conf=model_conf,
                messages=messages,
                temperature=temperature,
                stream=stream,
            )
            latency = time.time() - start_time
            logger.debug(f"模型 '{model_name}' 响应延迟: {latency:.2f}s")
            return content, usage

        except Exception as e:
            latency = time.time() - start_time
            error_str = str(e).lower()

            # 判断是否应该重试
            if self._should_failover(e):
                logger.warning(
                    f"模型 '{model_name}' 触发故障切换条件，错误: {e}"
                )
                raise  # tenacity 会捕获并根据 retry 条件重试

            # 不可重试的错误直接抛出
            logger.error(f"模型 '{model_name}' 不可重试的错误: {e}")
            raise

    def _should_failover(self, error: Exception) -> bool:
        """
        判断是否应触发故障切换。

        规则：
        - API 限流 (429) → 等待后重试（不切换）
        - 超时 → 切换模型
        - 认证错误 (401) → 不重试，直接切换
        - 服务器错误 (5xx) → 重试后切换
        - 网络错误 → 重试后切换
        """
        error_str = str(error).lower()

        # 认证错误：不重试，直接切换
        if "401" in error_str or "unauthorized" in error_str or "authentication" in error_str:
            return True

        # 超时错误：直接切换
        if "timeout" in error_str or "timed out" in error_str:
            return True

        # 其他都允许 tenacity 重试
        return True

    # --------------------------------------------------------
    # 记录与日志
    # --------------------------------------------------------

    def _build_record(
        self,
        task_type: str,
        model_name: str,
        model_conf: ModelConfig,
        usage: Dict[str, int],
        success: bool,
        error: str = "",
    ) -> LLMCallRecord:
        """构建调用记录"""
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        # 计算成本
        pricing = model_conf.pricing
        cost = (
            input_tokens / 1000 * pricing.input_per_1k
            + output_tokens / 1000 * pricing.output_per_1k
        )

        return LLMCallRecord(
            task_type=task_type,
            model_used=model_name,
            provider=model_conf.provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            success=success,
            error_message=error,
        )

    def _log_call(self, record: LLMCallRecord) -> None:
        """将调用记录追加到日志文件（JSON Lines 格式）"""
        try:
            record_dict = {
                "task_type": record.task_type,
                "model_used": record.model_used,
                "provider": record.provider,
                "input_tokens": record.input_tokens,
                "output_tokens": record.output_tokens,
                "total_tokens": record.input_tokens + record.output_tokens,
                "cost_usd": round(record.cost, 6),
                "success": record.success,
                "timestamp": record.timestamp,
                "latency_seconds": round(record.latency_seconds, 3),
                "error_message": record.error_message,
            }

            with open(self._call_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record_dict, ensure_ascii=False) + "\n")

        except Exception as e:
            logger.error(f"写入调用日志失败: {e}")

    # --------------------------------------------------------
    # 公开工具方法
    # --------------------------------------------------------

    def get_cost_summary(self) -> Dict[str, Any]:
        """返回成本汇总（委托给 CostTracker）"""
        return self.cost_tracker.get_summary()

    def reset_cost_tracker(self) -> None:
        """重置成本追踪"""
        self.cost_tracker.reset()
        logger.info("成本追踪已重置")

    def get_available_models(self) -> List[str]:
        """返回所有可用模型名"""
        return list(self.config.models.keys())

    def print_cost_summary(self) -> None:
        """打印格式化的成本汇总"""
        summary = self.get_cost_summary()
        print("\n" + "=" * 50)
        print("  LLM 成本汇总")
        print("=" * 50)
        print(f"  总成本:        ${summary['total_cost_usd']:.6f}")
        print(f"  总调用次数:     {summary['total_calls']}")
        print(f"  成功:          {summary['successful_calls']}")
        print(f"  失败:          {summary['failed_calls']}")
        print("-" * 50)
        for model, stats in summary.get("per_model", {}).items():
            print(f"  [{model}]")
            print(f"    成本:        ${stats['cost_usd']:.6f}")
            print(f"    input:       {stats['input_tokens']} tokens")
            print(f"    output:      {stats['output_tokens']} tokens")
        print("=" * 50 + "\n")


# ============================================================
# 便捷函数
# ============================================================

# 全局客户端实例（单例模式，按需创建）
_global_client: Optional[LLMClient] = None


def get_client(config_path: str = "config.yaml") -> LLMClient:
    """获取全局 LLMClient 实例（懒加载单例）"""
    global _global_client
    if _global_client is None:
        _global_client = LLMClient(config_path)
    return _global_client


def reset_global_client() -> None:
    """重置全局客户端"""
    global _global_client
    _global_client = None


# ============================================================
# 自测入口
# ============================================================

if __name__ == "__main__":
    # 简单自测：不发送真实 API 请求，仅验证初始化和配置
    print("=" * 50)
    print("  LLM 客户端自测")
    print("=" * 50)

    # 设置日志
    config = load_config()
    try:
        from scripts.utils import setup_logging
    except (ModuleNotFoundError, ImportError):
        from utils import setup_logging
    setup_logging(config.get("logging", {}))

    # 初始化客户端
    try:
        client = LLMClient()
        print(f"✅ 客户端初始化成功")
        print(f"   默认模型: {client.config.default_provider}")
        print(f"   可用模型: {client.get_available_models()}")
        print(f"   任务路由:")
        for task, model in client.config.task_model_mapping.items():
            print(f"     {task} → {model}")

        print(f"\n   故障切换阈值: {client.config.failover_threshold} 次连续失败")
        print(f"   全局最大重试: {client.config.global_max_retries}")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
