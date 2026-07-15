"""
搜索策略讨论器（Search Strategist）

功能：
- 基于你的学术档案，DeepSeek 自动提出初始搜索计划
- 支持自然语言交互修改（"地区改成美国+瑞士"）
- 美观的终端显示，带分隔线和状态指示
- 讨论历史记录到 logs/discussion_log.md
- 搜索计划保存到 checkpoints/ (JSON)
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from scripts.llm_client import LLMClient, TaskType
from scripts.utils import save_json, ensure_directory, get_timestamp

logger = logging.getLogger(__name__)

# ============================================================
# 常量
# ============================================================

SEPARATOR = "─" * 60
DOUBLE_SEP = "═" * 60
ARROW = "▸"


# ============================================================
# 搜索策略讨论器
# ============================================================

class SearchStrategist:
    """
    交互式搜索策略讨论器。

    使用示例:
        from scripts.profile_parser import ProfileParser
        from scripts.llm_client import LLMClient

        parser = ProfileParser()
        client = LLMClient()
        strategist = SearchStrategist(parser, client)
        plan = strategist.start_discussion()
    """

    def __init__(self, profile_parser, llm_client: LLMClient):
        """
        Args:
            profile_parser: ProfileParser 实例
            llm_client: LLMClient 实例
        """
        self.profile = profile_parser
        self.llm = llm_client

        # 讨论历史
        self.discussion_history: List[Dict[str, str]] = []
        self._round_id = get_timestamp()

        # 日志文件
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        """确保输出目录存在"""
        ensure_directory("checkpoints")
        ensure_directory("logs")

    # --------------------------------------------------------
    # 主入口
    # --------------------------------------------------------

    def start_discussion(self) -> Dict[str, Any]:
        """
        启动交互式搜索策略讨论。

        Returns:
            最终的搜索计划字典
        """
        print(f"\n{DOUBLE_SEP}")
        print("  🎯 搜索策略讨论器")
        print(f"  档案: {self.profile.profile.personal_info.name}")
        print(f"  轮次: {self._round_id}")
        print(f"{DOUBLE_SEP}")

        # 记录讨论开始
        self._log_discussion("system", "搜索策略讨论开始")
        self._log_discussion(
            "system",
            f"档案摘要: {self.profile.get_short_summary(max_length=200)}",
        )

        # 第一步：生成初始搜索计划
        print(f"\n  ⏳ DeepSeek 正在分析你的档案并生成初始搜索计划...\n")
        plan = self._generate_initial_plan()
        self._log_discussion("deepseek", f"初始计划: {json.dumps(plan, ensure_ascii=False, indent=2)}")

        # 第二步：交互循环
        print(f"\n  📋 DeepSeek 建议的初始搜索计划:\n")
        self._display_plan(plan)

        print(f"\n  💡 你可以用自然语言修改任何参数，例如：")
        print(f"     {ARROW} '地区改成只搜美国和瑞士'")
        print(f"     {ARROW} '关键词增加 AI for Science'")
        print(f"     {ARROW} '只要 Assistant Professor'")
        print(f"     {ARROW} '时间范围扩大到近5年'")
        print(f"     {ARROW} '确认' 或 'go' 提交当前计划")

        while True:
            user_input = input(f"\n  ✏️  你的修改 (或 '确认'/'go' 提交): ").strip()

            if not user_input:
                continue

            # 确认提交
            if user_input.lower() in ("确认", "go", "yes", "y", "ok", "done", "提交"):
                self._log_discussion("user", "确认提交")
                break

            # 显示帮助
            if user_input.lower() in ("?", "help", "帮助"):
                self._print_help()
                continue

            # 查看当前计划
            if user_input.lower() in ("show", "查看", "plan"):
                print()
                self._display_plan(plan)
                continue

            # 调用 LLM 处理修改
            self._log_discussion("user", user_input)
            print(f"\n  ⏳ DeepSeek 正在处理你的修改...")
            plan = self._process_modification(plan, user_input)
            self._log_discussion("deepseek", f"修改后计划: {json.dumps(plan, ensure_ascii=False, indent=2)}")

            print(f"\n  📋 更新后的搜索计划:\n")
            self._display_plan(plan)

        # 第三步：保存计划
        print(f"\n  ⏳ 正在保存搜索计划...")
        self._save_plan(plan)
        self._log_discussion("system", "讨论结束，计划已保存")
        self._save_discussion_log()

        print(f"\n{DOUBLE_SEP}")
        print(f"  ✅ 搜索计划已保存!")
        print(f"  📁 checkpoints/search_plan_{self._round_id}.json")
        print(f"  📝 讨论日志: logs/discussion_log_{self._round_id}.md")
        print(f"{DOUBLE_SEP}\n")

        return plan

    # --------------------------------------------------------
    # LLM 调用
    # --------------------------------------------------------

    def _generate_initial_plan(self) -> Dict[str, Any]:
        """
        调用 DeepSeek 基于档案生成初始搜索计划。

        Returns:
            搜索计划字典
        """
        profile_text = self.profile.get_profile_summary()
        keywords = self.profile.get_research_keywords()
        prefs = self.profile.profile.target_preferences

        prompt = f"""你是一个学术申请顾问。基于以下申请人档案，生成一个初始的教授搜索计划。

## 申请人档案
{profile_text}

## 要求
请以 JSON 格式输出搜索计划（不要包含 markdown 代码块标记，只输出纯 JSON）：

{{
    "regions": ["目标国家/地区列表"],
    "school_ranks": ["排名范围，如 Top 50"],
    "professor_ranks": ["目标教授职称列表"],
    "keywords": ["搜索关键词列表，10-15个，包含同义词"],
    "time_range_years": 年份数字,
    "max_results_per_query": 数字,
    "rationale": "简短说明为什么选择这些参数（2-3句话）"
}}

注意事项：
- regions 从申请人的目标地区中选择，不要超出范围
- professor_ranks 优先 Assistant Professor（最可能招人）
- keywords 要覆盖申请人所有研究兴趣方向
- time_range_years 建议3年
- 输出必须是合法的 JSON"""

        response = self.llm.call(
            messages=[{"role": "user", "content": prompt}],
            task_type=TaskType.SEARCH_DISCUSSION,
        )

        # 解析 JSON 响应
        try:
            # 清理可能的 markdown 代码块标记
            cleaned = response.strip()
            if cleaned.startswith("```"):
                # 移除 ```json ... ``` 包装
                lines = cleaned.split("\n")
                cleaned = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else cleaned
                cleaned = cleaned.strip()
            plan = json.loads(cleaned)
            plan["search_round_id"] = self._round_id
            plan["generated_at"] = datetime.now().isoformat()
            return plan
        except json.JSONDecodeError as e:
            logger.warning(f"JSON 解析失败，使用回退计划: {e}")
            return self._fallback_plan()

    def _process_modification(
        self, current_plan: Dict, user_input: str
    ) -> Dict[str, Any]:
        """
        调用 DeepSeek 根据用户输入修改搜索计划。

        Args:
            current_plan: 当前搜索计划
            user_input: 用户的自然语言修改请求

        Returns:
            修改后的搜索计划
        """
        prompt = f"""你是一个学术申请顾问。用户要求修改当前的教授搜索计划。

## 当前计划
```json
{json.dumps(current_plan, ensure_ascii=False, indent=2)}
```

## 用户修改请求
{user_input}

## 要求
根据用户的修改请求更新计划，并以 JSON 格式输出修改后的完整计划。
注意：
- 只修改用户提到的部分，其余保持不变
- 输出纯 JSON，不要包含 markdown 代码块标记
- 如果用户的请求不明确，做出合理推断
- 保持计划结构不变"""

        response = self.llm.call(
            messages=[{"role": "user", "content": prompt}],
            task_type=TaskType.SEARCH_DISCUSSION,
        )

        try:
            cleaned = response.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                cleaned = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else cleaned
                cleaned = cleaned.strip()
            new_plan = json.loads(cleaned)
            # 保留元信息
            new_plan["search_round_id"] = current_plan.get("search_round_id", self._round_id)
            new_plan["generated_at"] = datetime.now().isoformat()
            return new_plan
        except json.JSONDecodeError as e:
            logger.warning(f"修改响应 JSON 解析失败: {e}")
            print(f"\n  ⚠️  无法解析 DeepSeek 的响应，保留当前计划。")
            return current_plan

    # --------------------------------------------------------
    # 显示
    # --------------------------------------------------------

    def _display_plan(self, plan: Dict) -> None:
        """美化显示搜索计划"""
        regions = plan.get("regions", [])
        ranks = plan.get("school_ranks", [])
        prof_ranks = plan.get("professor_ranks", [])
        keywords = plan.get("keywords", [])
        time_range = plan.get("time_range_years", 3)
        max_results = plan.get("max_results_per_query", 50)
        rationale = plan.get("rationale", "")

        print(f"  {SEPARATOR}")
        print(f"  📍 地区:      {', '.join(regions)}")
        print(f"  🏫 学校排名:   {', '.join(ranks)}")
        print(f"  👨‍🏫 教授职称:   {', '.join(prof_ranks)}")
        print(f"  🔑 关键词 ({len(keywords)}个):")
        # 关键词分两行显示
        kw_lines = _wrap_keywords(keywords, max_width=50)
        for line in kw_lines:
            print(f"     {line}")
        print(f"  📅 时间范围:   近 {time_range} 年")
        print(f"  🔢 每查询上限: {max_results} 条")
        if rationale:
            print(f"  💬 理由: {rationale}")
        print(f"  {SEPARATOR}")

    @staticmethod
    def _print_help() -> None:
        """打印帮助信息"""
        print(f"""
  📖 修改命令示例:
     {ARROW} '地区改成 美国、瑞士、英国'
     {ARROW} '只要美国和加拿大'
     {ARROW} '关键词增加 natural language processing'
     {ARROW} '去掉 reinforcement learning'
     {ARROW} '只要 Assistant Professor'
     {ARROW} '时间改成近5年'
     {ARROW} '每查询最多30条结果'
     {ARROW} '学校排名改成 Top 30'

  其他命令:
     {ARROW} 'show' / '查看' — 重新显示当前计划
     {ARROW} 'help' / '?' — 显示此帮助
     {ARROW} 'go' / '确认' — 提交并保存
""")

    # --------------------------------------------------------
    # 记录与保存
    # --------------------------------------------------------

    def _save_plan(self, plan: Dict) -> None:
        """保存搜索计划到 checkpoints/"""
        filename = f"checkpoints/search_plan_{self._round_id}.json"
        save_json(plan, filename)
        logger.info(f"搜索计划已保存: {filename}")

    def _log_discussion(self, role: str, content: str) -> None:
        """记录讨论历史"""
        self.discussion_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })

    def _save_discussion_log(self) -> None:
        """将讨论历史保存为 Markdown 文件"""
        filename = f"logs/discussion_log_{self._round_id}.md"

        lines = [
            f"# 搜索策略讨论日志",
            f"",
            f"**轮次**: {self._round_id}",
            f"**时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**申请人**: {self.profile.profile.personal_info.name}",
            f"",
            f"---",
            f"",
        ]

        for entry in self.discussion_history:
            role_label = {
                "system": "🔧 系统",
                "user": "👤 用户",
                "deepseek": "🤖 DeepSeek",
            }.get(entry["role"], entry["role"])

            ts = entry["timestamp"][:19]
            lines.append(f"### {role_label} ({ts})")
            lines.append(f"")
            lines.append(entry["content"])
            lines.append(f"")

        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(f"讨论日志已保存: {filename}")

    # --------------------------------------------------------
    # 回退方案
    # --------------------------------------------------------

    def _fallback_plan(self) -> Dict[str, Any]:
        """当 LLM 调用失败时的回退搜索计划"""
        prefs = self.profile.profile.target_preferences
        keywords = self.profile.get_research_keywords()

        return {
            "regions": prefs.locations,
            "school_ranks": prefs.school_ranks,
            "professor_ranks": prefs.professor_ranks,
            "keywords": keywords[:15],
            "time_range_years": 3,
            "max_results_per_query": 50,
            "rationale": "基于档案自动生成（回退方案）",
            "search_round_id": self._round_id,
            "generated_at": datetime.now().isoformat(),
        }


# ============================================================
# 工具函数
# ============================================================

def _wrap_keywords(keywords: List[str], max_width: int = 55) -> List[str]:
    """将关键词列表格式化为多行显示"""
    lines = []
    current = ""
    for kw in keywords:
        candidate = f"{current}, {kw}" if current else kw
        if len(candidate) > max_width and current:
            lines.append(current)
            current = kw
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


# ============================================================
# 非交互模式（用于自动化流水线）
# ============================================================

def generate_plan_auto(
    profile_parser,
    llm_client: LLMClient,
    custom_modifications: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    非交互模式：自动生成搜索计划。

    适用于无人值守的批量运行场景。

    Args:
        profile_parser: ProfileParser 实例
        llm_client: LLMClient 实例
        custom_modifications: 预设的修改列表，如 ["地区改成美国和瑞士"]

    Returns:
        最终搜索计划
    """
    strategist = SearchStrategist(profile_parser, llm_client)

    # 生成初始计划
    print("⏳ 生成初始搜索计划...")
    plan = strategist._generate_initial_plan()

    # 应用预设修改
    if custom_modifications:
        for mod in custom_modifications:
            print(f"  应用修改: {mod}")
            plan = strategist._process_modification(plan, mod)

    # 保存
    strategist._save_plan(plan)
    print(f"✅ 计划已保存: checkpoints/search_plan_{strategist._round_id}.json")

    return plan


# ============================================================
# 自测入口
# ============================================================

if __name__ == "__main__":
    from scripts.profile_parser import ProfileParser

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    print(f"\n{DOUBLE_SEP}")
    print("  搜索策略讨论器 - 自测 (非交互模式)")
    print(f"{DOUBLE_SEP}")

    # 加载组件
    parser = ProfileParser("profiles/my_profile_template.json")
    client = LLMClient()

    # 非交互模式测试
    print("\n[1] 自动生成搜索计划...")
    plan = generate_plan_auto(
        parser,
        client,
        custom_modifications=["地区改成美国、加拿大、瑞士、英国"],
    )

    print(f"\n[2] 生成的搜索计划:")
    print(f"   地区: {plan.get('regions')}")
    print(f"   职称: {plan.get('professor_ranks')}")
    print(f"   关键词: {plan.get('keywords', [])[:5]}...")
    print(f"   时间: 近{plan.get('time_range_years')}年")
    print(f"   理由: {plan.get('rationale', '')[:80]}")

    print(f"\n✅ 自测完成")
