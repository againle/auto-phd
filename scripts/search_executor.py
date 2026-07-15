"""
搜索执行器（Search Executor）

功能：
- 接收搜索计划，依次执行关键词搜索
- 自动去重（基于 scholar_id）
- 为每位教授创建独立文件夹（info.json + status.json）
- 断点续传（从 checkpoints 恢复）
- tqdm 进度条 + 搜索摘要报告

文件夹结构:
professors/{institution}_{name}/
├── info.json       # 教授详细信息
├── papers/         # 后续下载的论文
└── status.json     # 当前处理状态
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Set

from tqdm import tqdm

from scripts.scholar_api import search_professors, batch_search_professors
from scripts.professor_scorer import ProfessorScorer
from scripts.utils import (
    ensure_directory, save_json, load_json,
    get_timestamp, safe_filename, load_config,
)

logger = logging.getLogger(__name__)

# ============================================================
# 常量
# ============================================================

STATUS_ORDER = [
    "pending_score",       # 待评分
    "scored",              # 已评分
    "ready_for_reading",   # 可开始文献阅读
    "reading",             # 正在阅读论文
    "ready_for_email",     # 可发送邮件
    "email_drafted",       # 邮件已生成
    "sent",                # 已发送
    "replied",             # 已回复
    "rejected",            # 被拒/不匹配
    "skipped",             # 跳过
]

CHECKPOINT_INTERVAL = 10  # 每找到10个教授保存一次检查点


# ============================================================
# 搜索执行器
# ============================================================

class SearchExecutor:
    """
    搜索执行器。

    使用示例:
        executor = SearchExecutor(scorer, config)
        professors = executor.execute_plan(plan)
    """

    def __init__(
        self,
        scorer: Optional[ProfessorScorer] = None,
        config_path: str = "config.yaml",
    ):
        """
        Args:
            scorer: ProfessorScorer 实例（可选，用于质量筛选）
            config_path: 配置文件路径
        """
        self.config = load_config(config_path)
        self.scorer = scorer

        # 搜索配置
        search_conf = self.config.get("search", {})
        self.max_results = search_conf.get("max_results_per_query", 50)
        self.min_pub_years = search_conf.get("min_publication_years", 3)
        self.api_delay = search_conf.get("api_delay_seconds", 1.0)

        # 质量配置
        quality_conf = self.config.get("quality", {})
        self.min_publications = quality_conf.get("min_publication_count", 3)

        # 熔断器
        cb = self.config.get("circuit_breaker", {})
        self.max_api_calls = cb.get("max_api_calls_per_run", 500)
        self.max_search_attempts = cb.get("max_search_attempts", 200)
        self.stagnation_threshold = cb.get("stagnation_threshold", 3)

        # 统计
        self._api_call_count = 0
        self._round_id = get_timestamp()
        self._checkpoint_file = f"checkpoints/executor_{self._round_id}.json"

        ensure_directory("professors")
        ensure_directory("checkpoints")

        logger.info(f"搜索执行器初始化 (round={self._round_id})")

    # --------------------------------------------------------
    # 主入口
    # --------------------------------------------------------

    def execute_plan(
        self,
        plan: Dict[str, Any],
        resume: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        执行搜索计划。

        Args:
            plan: 搜索计划（来自 SearchStrategist）
            resume: 是否从检查点恢复

        Returns:
            找到的教授列表（已去重）
        """
        plan_round = plan.get("search_round_id", self._round_id)
        keywords = plan.get("keywords", [])
        regions = plan.get("regions", [])
        time_range = plan.get("time_range_years", self.min_pub_years)
        max_per = plan.get("max_results_per_query", self.max_results)

        print(f"\n{'═' * 60}")
        print(f"  🔍 开始执行搜索计划")
        print(f"  {'─' * 60}")
        print(f"  轮次: {plan_round}")
        print(f"  关键词: {len(keywords)} 个")
        print(f"  地区: {', '.join(regions)}")
        print(f"  时间: 近 {time_range} 年")
        print(f"  {'═' * 60}\n")

        # 尝试从检查点恢复
        all_professors: List[Dict] = []
        searched_keywords: Set[str] = set()
        total_before_dedup = 0

        if resume:
            checkpoint = self._load_checkpoint()
            if checkpoint:
                all_professors = checkpoint.get("professors", [])
                searched_keywords = set(checkpoint.get("searched_keywords", []))
                total_before_dedup = checkpoint.get("total_before_dedup", 0)
                logger.info(
                    f"从检查点恢复: {len(all_professors)} 位教授, "
                    f"已完成 {len(searched_keywords)}/{len(keywords)} 个关键词"
                )

        # 逐关键词搜索
        remaining = [kw for kw in keywords if kw not in searched_keywords]
        stagnation_count = 0
        prev_count = len(all_professors)

        with tqdm(total=len(remaining), desc="搜索进度", unit="query", ncols=80) as pbar:
            for idx, keyword in enumerate(remaining):
                # 熔断检查
                if self._api_call_count >= self.max_api_calls:
                    logger.warning(f"达到 API 调用上限 ({self.max_api_calls})，停止搜索")
                    pbar.write(f"⚠️  API 调用上限已达，停止搜索。")
                    break

                if len(all_professors) >= self.max_search_attempts:
                    logger.warning(f"达到搜索尝试上限 ({self.max_search_attempts})")
                    pbar.write(f"⚠️  搜索尝试上限已达 ({self.max_search_attempts})。")
                    break

                # 搜索
                pbar.set_postfix_str(f"搜索: {keyword[:25]}...")
                try:
                    results = search_professors(
                        keyword,
                        limit=max_per,
                        year=time_range,
                    )
                    self._api_call_count += 1
                    total_before_dedup += len(results)
                except Exception as e:
                    logger.error(f"关键词 '{keyword}' 搜索失败: {e}")
                    pbar.write(f"  ❌ 搜索失败: {keyword[:30]}... ({e})")
                    results = []

                # 去重 & 合并
                new_count = self._merge_and_deduplicate(all_professors, results)

                # 停滞检测
                if len(all_professors) - prev_count == 0:
                    stagnation_count += 1
                else:
                    stagnation_count = 0
                prev_count = len(all_professors)

                if stagnation_count >= self.stagnation_threshold:
                    pbar.write(
                        f"⚠️  连续 {self.stagnation_threshold} 个查询无新增教授，"
                        f"提前停止。"
                    )
                    break

                searched_keywords.add(keyword)
                pbar.update(1)

                # 定期保存检查点
                if (idx + 1) % CHECKPOINT_INTERVAL == 0:
                    self._save_checkpoint(all_professors, searched_keywords, total_before_dedup)
                    pbar.write(f"  💾 检查点已保存 ({len(all_professors)} 位教授)")

                time.sleep(self.api_delay * 0.3)

        # 最终保存
        self._save_checkpoint(all_professors, searched_keywords, total_before_dedup)

        # 为每位教授创建文件夹
        print(f"\n  📁 创建教授文件夹...")
        created = 0
        for prof in tqdm(all_professors, desc="创建文件夹", unit="prof", ncols=80):
            try:
                self._create_professor_folder(prof, plan_round)
                created += 1
            except Exception as e:
                logger.error(f"创建文件夹失败 {prof.get('name', '?')}: {e}")

        # 生成摘要报告
        self._generate_summary(all_professors, plan, total_before_dedup)

        print(f"\n{'═' * 60}")
        print(f"  ✅ 搜索执行完成")
        print(f"  {'─' * 60}")
        print(f"  原始结果:    {total_before_dedup} 条")
        print(f"  去重后:      {len(all_professors)} 位教授")
        print(f"  创建文件夹:  {created}")
        print(f"  API调用次数: {self._api_call_count}")
        print(f"  {'═' * 60}\n")

        return all_professors

    # --------------------------------------------------------
    # 核心操作
    # --------------------------------------------------------

    def _merge_and_deduplicate(
        self,
        existing: List[Dict],
        newcomers: List[Dict],
    ) -> int:
        """
        合并新结果到已有列表，按 scholar_id 去重。

        Returns:
            新增的教授数量
        """
        seen_ids = {p.get("scholar_id", "") for p in existing}
        new_count = 0

        for prof in newcomers:
            sid = prof.get("scholar_id", "")
            if sid and sid not in seen_ids:
                existing.append(prof)
                seen_ids.add(sid)
                new_count += 1

        if new_count > 0:
            logger.debug(f"去重合并: +{new_count}, 总计 {len(existing)}")

        return new_count

    def _create_professor_folder(
        self, professor: Dict, plan_round: str = ""
    ) -> Path:
        """
        为教授创建独立文件夹和 info.json / status.json。

        Args:
            professor: 教授数据字典
            plan_round: 搜索轮次标识

        Returns:
            文件夹 Path
        """
        name = professor.get("name", "unknown")
        institution = professor.get("institution", "unknown")

        # 安全文件夹名
        folder_name = safe_filename(f"{institution}_{name}")[:80]
        prof_dir = Path(f"professors/{folder_name}")
        ensure_directory(str(prof_dir))
        ensure_directory(str(prof_dir / "papers"))

        # 评分（如果有 scorer）
        quality_score = 0
        if self.scorer:
            score_result = self.scorer.score(professor)
            quality_score = score_result["total_score"]
            professor["_score"] = quality_score
            professor["_breakdown"] = score_result["breakdown"]
            professor["_details"] = score_result["details"]

        # 构建 info.json
        info = {
            "professor_id": professor.get("scholar_id", ""),
            "name": name,
            "institution": institution,
            "email": professor.get("email", ""),
            "rank": professor.get("rank", self._infer_rank_from_data(professor)),
            "research_topics": professor.get("research_topics", []),
            "publication_count": professor.get("publication_count", 0),
            "h_index": professor.get("h_index", 0),
            "citation_count": professor.get("citation_count", 0),
            "recent_papers": professor.get("recent_papers", []),
            "quality_score": quality_score,
            "score_breakdown": professor.get("_breakdown", {}),
            "match_details": professor.get("_details", {}),
            "status": "pending_score" if quality_score == 0 else "scored",
            "found_timestamp": datetime.now().isoformat(),
            "search_round": plan_round,
            "scholar_url": professor.get("url", ""),
        }

        save_json(info, str(prof_dir / "info.json"))

        # 构建 status.json
        status = {
            "current_status": info["status"],
            "status_history": [
                {
                    "status": info["status"],
                    "timestamp": datetime.now().isoformat(),
                    "note": "初始发现",
                }
            ],
            "last_updated": datetime.now().isoformat(),
        }

        save_json(status, str(prof_dir / "status.json"))

        logger.debug(f"文件夹已创建: {folder_name}")
        return prof_dir

    # --------------------------------------------------------
    # 检查点
    # --------------------------------------------------------

    def _save_checkpoint(
        self,
        professors: List[Dict],
        searched_keywords: Set[str],
        total_before_dedup: int = 0,
    ) -> None:
        """保存搜索检查点"""
        checkpoint = {
            "round_id": self._round_id,
            "saved_at": datetime.now().isoformat(),
            "professor_count": len(professors),
            "total_before_dedup": total_before_dedup,
            "api_call_count": self._api_call_count,
            "searched_keywords": list(searched_keywords),
            "professors": professors,
        }
        save_json(checkpoint, self._checkpoint_file)

    def _load_checkpoint(self) -> Optional[Dict]:
        """加载上次的搜索检查点"""
        # 优先当前轮次，否则找最新的
        checkpoint_path = Path(self._checkpoint_file)
        if not checkpoint_path.exists():
            # 找最新的检查点文件
            checkpoints = sorted(
                Path("checkpoints").glob("executor_*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if checkpoints:
                checkpoint_path = checkpoints[0]
                logger.info(f"加载最新检查点: {checkpoint_path.name}")
            else:
                return None

        try:
            return load_json(str(checkpoint_path))
        except Exception as e:
            logger.warning(f"加载检查点失败: {e}")
            return None

    # --------------------------------------------------------
    # 摘要报告
    # --------------------------------------------------------

    def _generate_summary(
        self,
        professors: List[Dict],
        plan: Dict,
        total_before_dedup: int = 0,
    ) -> None:
        """生成 Markdown 搜索摘要报告"""
        # 统计
        if professors:
            scores = [p.get("_score", p.get("quality_score", 0)) for p in professors]
            avg_score = sum(scores) / len(scores) if scores else 0
            max_score = max(scores) if scores else 0
            min_score = min(scores) if scores else 0
            h_indices = [p.get("h_index", 0) for p in professors]
            avg_h = sum(h_indices) / len(h_indices) if h_indices else 0
        else:
            avg_score = max_score = min_score = avg_h = 0

        # 去重后筛选通过阈值的
        if self.scorer:
            qualified = [
                p for p in professors
                if p.get("_score", 0) >= self.config.get("quality", {}).get("min_quality_score", 70)
            ]
        else:
            qualified = professors

        report = f"""# 搜索摘要报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**搜索轮次**: {self._round_id}

---

## 搜索参数

| 参数 | 值 |
|------|-----|
| 关键词数量 | {len(plan.get('keywords', []))} |
| 目标地区 | {', '.join(plan.get('regions', []))} |
| 学校排名 | {', '.join(plan.get('school_ranks', []))} |
| 教授职称 | {', '.join(plan.get('professor_ranks', []))} |
| 时间范围 | 近 {plan.get('time_range_years', 3)} 年 |
| 每查询上限 | {plan.get('max_results_per_query', 50)} |

## 搜索结果

| 指标 | 值 |
|------|-----|
| 原始结果 | {total_before_dedup} |
| 去重后 | {len(professors)} |
| 通过质量筛选 | {len(qualified)} |
| API 调用次数 | {self._api_call_count} |

## 质量统计

| 指标 | 值 |
|------|-----|
| 平均质量分 | {avg_score:.1f} |
| 最高分 | {max_score} |
| 最低分 | {min_score} |
| 平均 h-index | {avg_h:.1f} |

## Top 10 教授

| # | 姓名 | 机构 | h-index | 论文数 | 得分 |
|---|------|------|---------|--------|------|
"""

        # Top 10
        sorted_profs = sorted(
            professors,
            key=lambda p: p.get("_score", p.get("quality_score", 0)),
            reverse=True,
        )
        for i, prof in enumerate(sorted_profs[:10], 1):
            score = prof.get("_score", prof.get("quality_score", 0))
            report += (
                f"| {i} | {prof.get('name', '?')[:25]} "
                f"| {prof.get('institution', '?')[:25]} "
                f"| {prof.get('h_index', 0)} "
                f"| {prof.get('publication_count', 0)} "
                f"| {score} |\n"
            )

        report += f"""
---

*报告由 SearchExecutor 自动生成*
"""

        # 保存
        report_path = f"logs/search_summary_{self._round_id}.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"摘要报告已保存: {report_path}")
        print(f"\n  📊 摘要报告: {report_path}")

    # --------------------------------------------------------
    # 辅助
    # --------------------------------------------------------

    @staticmethod
    def _infer_rank_from_data(professor: Dict) -> str:
        """从数据推断职称"""
        h_index = professor.get("h_index", 0)
        if h_index <= 12:
            return "Assistant Professor (inferred)"
        elif h_index <= 25:
            return "Associate Professor (inferred)"
        else:
            return "Professor (inferred)"

    def get_professor_status(self, prof_dir: str) -> Optional[Dict]:
        """读取教授文件夹中的 status.json"""
        status_path = Path(prof_dir) / "status.json"
        if status_path.exists():
            return load_json(str(status_path))
        return None

    def update_professor_status(
        self, prof_dir: str, new_status: str, note: str = ""
    ) -> None:
        """更新教授状态"""
        status_path = Path(prof_dir) / "status.json"
        status = load_json(str(status_path)) if status_path.exists() else {}

        status["current_status"] = new_status
        status["last_updated"] = datetime.now().isoformat()

        history = status.get("status_history", [])
        history.append({
            "status": new_status,
            "timestamp": datetime.now().isoformat(),
            "note": note,
        })
        status["status_history"] = history

        save_json(status, str(status_path))


# ============================================================
# 自测入口
# ============================================================

if __name__ == "__main__":
    from scripts.profile_parser import ProfileParser
    from scripts.professor_scorer import ProfessorScorer

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    print(f"\n{'═' * 60}")
    print("  搜索执行器 - 自测 (模拟数据)")
    print(f"{'═' * 60}")

    # 初始化
    parser = ProfileParser("profiles/my_profile_template.json")
    scorer = ProfessorScorer(parser)
    executor = SearchExecutor(scorer=scorer)

    # 模拟搜索计划
    mock_plan = {
        "search_round_id": get_timestamp(),
        "regions": ["United States", "Canada"],
        "school_ranks": ["Top 50"],
        "professor_ranks": ["Assistant Professor", "Associate Professor"],
        "keywords": ["deep learning biology", "protein structure prediction"],
        "time_range_years": 3,
        "max_results_per_query": 3,
    }

    # 模拟教授数据（替代真实 API 调用）
    mock_professors = [
        {
            "name": "Dr. Alice Researcher",
            "institution": "Stanford University",
            "research_topics": ["deep learning", "protein structure", "ai for science", "molecular dynamics"],
            "publication_count": 25,
            "h_index": 15,
            "citation_count": 1200,
            "recent_papers": [
                {"title": "Deep Learning for Protein Folding", "year": 2025, "citations": 50, "venue": "Nature Methods", "paper_id": "p1"},
                {"title": "AI in Structural Biology", "year": 2024, "citations": 35, "venue": "Science", "paper_id": "p2"},
            ],
            "scholar_id": "mock_001",
            "url": "https://semanticscholar.org/author/001",
        },
        {
            "name": "Prof. Bob Scientist",
            "institution": "MIT",
            "research_topics": ["reinforcement learning", "drug discovery", "healthcare ai"],
            "publication_count": 40,
            "h_index": 28,
            "citation_count": 5000,
            "recent_papers": [
                {"title": "RL for Drug Design", "year": 2025, "citations": 80, "venue": "ICML", "paper_id": "p3"},
            ],
            "scholar_id": "mock_002",
            "url": "https://semanticscholar.org/author/002",
        },
        {
            "name": "Dr. Alice Researcher",  # 重复：同一 scholar_id
            "institution": "Stanford University",
            "research_topics": ["deep learning", "ai"],
            "publication_count": 25,
            "h_index": 15,
            "citation_count": 1200,
            "recent_papers": [],
            "scholar_id": "mock_001",  # 相同 ID
            "url": "",
        },
    ]

    # 直接创建文件夹（跳过真实 API）
    print("\n[1] 创建教授文件夹...")
    for prof in mock_professors[:2]:  # 只用前2个（第3个是重复）
        folder = executor._create_professor_folder(prof, mock_plan["search_round_id"])
        print(f"    ✅ {folder}")

    # 测试去重
    print("\n[2] 去重测试...")
    existing = [mock_professors[0]]
    new_count = executor._merge_and_deduplicate(existing, [mock_professors[2]])
    print(f"    新增: {new_count}, 总计: {len(existing)} (应为 0, 1)")

    # 测试状态更新
    print("\n[3] 状态更新测试...")
    inst = mock_professors[0]["institution"]
    pname = mock_professors[0]["name"]
    prof_dir = "professors/" + safe_filename(f"{inst}_{pname}")[:80]
    executor.update_professor_status(prof_dir, "ready_for_email", "评分通过，准备发邮件")
    status = executor.get_professor_status(prof_dir)
    print(f"    当前状态: {status['current_status']}")
    print(f"    历史: {len(status['status_history'])} 条记录")

    # 生成摘要
    print("\n[4] 摘要报告...")
    executor._generate_summary(existing, mock_plan, total_before_dedup=3)

    # 列出教授文件夹
    print("\n[5] 教授文件夹:")
    for d in sorted(Path("professors").iterdir()):
        if d.is_dir():
            files = list(d.glob("*"))
            print(f"    {d.name}/ ({len(files)} 个文件)")

    print(f"\n✅ 自测完成")
