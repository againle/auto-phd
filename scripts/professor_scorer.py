"""
教授质量评分器

评分维度（满分100）：
1. 研究兴趣匹配度（40分）— 关键词 Jaccard 相似度
2. 学术产出（30分）— 近3年论文数量
3. 职称偏好（20分）— Assistant Prof 最高
4. 地理位置匹配（10分）— 是否在目标地区
"""

import logging
from typing import List, Dict, Any, Optional, Tuple

from scripts.utils import load_config

logger = logging.getLogger(__name__)

# ============================================================
# 常量
# ============================================================

# 顶级会议/期刊列表（用于学术产出加分）
TOP_VENUES = {
    "nature", "science", "cell",
    "neurips", "icml", "iclr", "aaai", "ijcai", "cvpr", "iccv", "eccv",
    "acl", "emnlp", "naacl",
    "sigmod", "vldb", "kdd", "www",
    "osdi", "sosp", "nsdi", "sigcomm", "mobicom",
    "isca", "micro", "hpc", "asplos",
    "pldi", "popl", "oopsla",
    "chi", "uist",
    "icra", "iros", "rss",
    "nature machine intelligence", "nature methods", "nature biotechnology",
    "ieee transactions on", "journal of machine learning research",
    "proceedings of the national academy of sciences",
    "bioinformatics", "plos computational biology",
}


# ============================================================
# 评分器
# ============================================================

class ProfessorScorer:
    """
    教授质量评分器。

    使用示例:
        parser = ProfileParser()
        scorer = ProfessorScorer(parser)
        result = scorer.score(professor_dict)
        print(result["total_score"])
    """

    def __init__(self, profile_parser, config_path: str = "config.yaml"):
        """
        Args:
            profile_parser: ProfileParser 实例
            config_path: 配置文件路径
        """
        self.profile = profile_parser
        self.config = load_config(config_path)

        # 阈值配置
        quality = self.config.get("quality", {})
        self.similarity_threshold = quality.get("similarity_threshold", 0.6)
        self.min_publication_count = quality.get("min_publication_count", 3)
        self.preferred_ranks = quality.get("preferred_ranks", [
            "Assistant Professor", "Associate Professor",
        ])

        # 目标地区
        target_prefs = self.profile.profile.target_preferences
        self.target_locations = [loc.lower() for loc in target_prefs.locations]

        # 我的研究关键词
        self.my_keywords = set(self.profile.get_research_keywords())

        logger.info(
            f"评分器初始化: {len(self.my_keywords)} 个关键词, "
            f"目标地区: {self.target_locations}, "
            f"偏好职称: {self.preferred_ranks}"
        )

    # --------------------------------------------------------
    # 主评分接口
    # --------------------------------------------------------

    def score(self, professor: Dict[str, Any]) -> Dict[str, Any]:
        """
        对单个教授进行多维度评分。

        Args:
            professor: 教授字典（来自 scholar_api.search_professors）

        Returns:
            评分结果字典:
            {
                "total_score": int,
                "breakdown": {dimension: score},
                "details": {...},
                "passed_threshold": bool,
            }
        """
        # 各维度独立评分
        research_score, research_details = self._score_research_match(professor)
        pub_score, pub_details = self._score_publications(professor)
        rank_score, rank_details = self._score_rank(professor)
        location_score, location_details = self._score_location(professor)

        breakdown = {
            "research_match": research_score,
            "publications": pub_score,
            "rank": rank_score,
            "location": location_score,
        }
        total = sum(breakdown.values())

        result = {
            "total_score": total,
            "breakdown": breakdown,
            "details": {
                **research_details,
                **pub_details,
                **rank_details,
                **location_details,
            },
            "passed_threshold": self._check_threshold(total, professor),
        }

        logger.debug(
            f"评分: {professor.get('name', '?')[:30]} → "
            f"{total}/100 (R:{research_score} P:{pub_score} Rk:{rank_score} L:{location_score})"
        )

        return result

    def batch_score(
        self,
        professors: List[Dict[str, Any]],
        sort: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        批量评分。

        Args:
            professors: 教授列表
            sort: 是否按总分降序排列

        Returns:
            评分后的教授列表，每个元素新增:
            - _score: 总分
            - _breakdown: 分项得分
            - _details: 评分细节
            - _passed: 是否通过阈值
        """
        logger.info(f"开始批量评分: {len(professors)} 位教授")

        for prof in professors:
            result = self.score(prof)
            prof["_score"] = result["total_score"]
            prof["_breakdown"] = result["breakdown"]
            prof["_details"] = result["details"]
            prof["_passed"] = result["passed_threshold"]

        if sort:
            professors.sort(key=lambda p: p["_score"], reverse=True)

        passed = sum(1 for p in professors if p.get("_passed", False))
        logger.info(
            f"批量评分完成: {passed}/{len(professors)} 位通过阈值 "
            f"(最高分: {professors[0]['_score'] if professors else 'N/A'})"
        )

        return professors

    # --------------------------------------------------------
    # 各维度评分
    # --------------------------------------------------------

    def _score_research_match(self, professor: Dict) -> Tuple[int, Dict]:
        """
        研究兴趣匹配度评分（满分40）。

        使用 Jaccard 相似度 × 40。
        """
        prof_topics = set(
            t.lower() for t in professor.get("research_topics", [])
        )

        similarity = self.calculate_similarity(
            list(self.my_keywords),
            list(prof_topics),
        )

        score = round(similarity * 40)

        # 找出匹配的关键词
        matched = list(self.my_keywords & prof_topics)

        details = {
            "matched_keywords": matched,
            "matched_count": len(matched),
            "similarity_score": round(similarity, 3),
            "your_keywords": list(self.my_keywords)[:10],
            "professor_topics": list(prof_topics)[:10],
        }

        return score, details

    def _score_publications(self, professor: Dict) -> Tuple[int, Dict]:
        """
        学术产出评分（满分30）。

        近3年论文数量:
        - >=10篇: 30分
        - 7-9篇: 25分
        - 5-6篇: 20分
        - 3-4篇: 15分
        - 1-2篇: 8分
        - 0篇: 0分

        顶会/顶刊加分: 最多额外+5（已纳入上限）
        最终上限30分。
        """
        recent_papers = professor.get("recent_papers", [])
        pub_count = len(recent_papers)

        # 基础分
        if pub_count >= 10:
            base_score = 30
        elif pub_count >= 7:
            base_score = 25
        elif pub_count >= 5:
            base_score = 20
        elif pub_count >= 3:
            base_score = 15
        elif pub_count >= 1:
            base_score = 8
        else:
            base_score = 0

        # 顶会/顶刊加分
        top_count = 0
        for paper in recent_papers:
            venue = (paper.get("venue", "") or "").lower()
            title = (paper.get("title", "") or "").lower()
            # 检查 venue 或 title 是否匹配顶会/刊
            combined = f"{venue} {title}"
            if any(tv in combined for tv in TOP_VENUES):
                top_count += 1

        # 顶会论文每篇额外+1，最多+5
        bonus = min(top_count, 5)
        score = min(base_score + bonus, 30)

        details = {
            "recent_publication_count": pub_count,
            "total_publication_count": professor.get("publication_count", 0),
            "h_index": professor.get("h_index", 0),
            "top_venue_count": top_count,
            "publication_bonus": bonus,
        }

        return score, details

    def _score_rank(self, professor: Dict) -> Tuple[int, Dict]:
        """
        职称偏好评分（满分20）。

        Assistant Professor: 20分（最可能招人）
        Associate Professor: 15分
        Professor: 10分
        未知: 5分
        """
        institution = (professor.get("institution", "") or "").lower()

        # 从机构信息推断职称（Semantic Scholar 不直接提供职称）
        # 策略：检查 institution 字符串中是否包含职称关键词
        rank = self._infer_rank(professor)

        rank_scores = {
            "assistant professor": 20,
            "associate professor": 15,
            "professor": 10,
            "unknown": 5,
        }

        score = rank_scores.get(rank, 5)

        details = {
            "inferred_rank": rank,
            "institution": professor.get("institution", "Unknown"),
        }

        return score, details

    def _score_location(self, professor: Dict) -> Tuple[int, Dict]:
        """
        地理位置匹配评分（满分10）。

        机构所在国家在目标地区列表内: 10分
        否则: 0分
        """
        institution = (professor.get("institution", "") or "").lower()
        location_matched = False
        matched_location = ""

        for target in self.target_locations:
            # 检查机构名称或地址是否包含目标地区
            if target in institution:
                location_matched = True
                matched_location = target
                break

        # 如果机构名没匹配到，尝试更宽松的匹配
        if not location_matched:
            country_map = {
                "united states": ["usa", "united states", "america", "u.s.", "u.s.a"],
                "canada": ["canada"],
                "united kingdom": ["uk", "united kingdom", "britain", "england"],
                "switzerland": ["switzerland", "eth", "epfl"],
                "germany": ["germany"],
                "france": ["france"],
                "australia": ["australia"],
                "china": ["china"],
                "japan": ["japan"],
                "singapore": ["singapore"],
                "europe": ["europe"],
                "asia": ["asia"],
            }
            for target_loc, aliases in country_map.items():
                if target_loc in self.target_locations or any(
                    alias in self.target_locations for alias in aliases
                ):
                    if any(alias in institution for alias in aliases):
                        location_matched = True
                        matched_location = target_loc
                        break

        score = 10 if location_matched else 0

        details = {
            "in_target_location": location_matched,
            "matched_location": matched_location,
            "target_locations": self.target_locations,
        }

        return score, details

    # --------------------------------------------------------
    # 辅助方法
    # --------------------------------------------------------

    def calculate_similarity(self, topics1: List[str], topics2: List[str]) -> float:
        """
        计算两个研究主题列表的 Jaccard 相似度（0-1）。

        Jaccard = |A ∩ B| / |A ∪ B|
        """
        set1 = set(t.lower() for t in topics1)
        set2 = set(t.lower() for t in topics2)

        if not set1 or not set2:
            return 0.0

        intersection = set1 & set2
        union = set1 | set2

        if not union:
            return 0.0

        return len(intersection) / len(union)

    def _infer_rank(self, professor: Dict) -> str:
        """
        从教授信息推断职称。

        Semantic Scholar 不直接提供职称，需要从机构/姓名等推断。
        这里提供一个基础版本，后续可通过学校官网爬虫增强。
        """
        institution = (professor.get("institution", "") or "").lower()
        name = (professor.get("name", "") or "").lower()

        # 简单启发式: 用 h-index 和被引数推断
        # 低 h-index (<10) + 有新论文 → 可能是 Assistant Professor
        # 中等 h-index (10-25) → 可能是 Associate
        # 高 h-index (>25) → 可能是 Full Professor
        h_index = professor.get("h_index", 0)
        pub_count = professor.get("publication_count", 0)
        recent_count = len(professor.get("recent_papers", []))

        # 如果机构名中直接有职称信息（很少见但有可能）
        if "assistant professor" in institution or "assistant prof" in institution:
            return "assistant professor"
        if "associate professor" in institution or "associate prof" in institution:
            return "associate professor"

        # 启发式推断
        if h_index <= 12 and recent_count >= 2:
            return "assistant professor"
        elif h_index <= 25:
            return "associate professor"
        elif h_index > 25:
            return "professor"

        return "unknown"

    def _check_threshold(self, total_score: int, professor: Dict) -> bool:
        """
        检查教授是否通过最低质量阈值。

        条件（需全部满足）:
        - 总分 >= 配置文件中的 min_quality_score（如果有）
        - 研究相似度 >= similarity_threshold
        - 发表数 >= min_publication_count
        """
        # 检查最低质量分
        min_quality = self.profile.profile.target_preferences.min_quality_score
        if total_score < min_quality:
            return False

        # 检查发表数
        pub_count = professor.get("publication_count", 0)
        if pub_count < self.min_publication_count:
            return False

        return True

    # --------------------------------------------------------
    # 报告生成
    # --------------------------------------------------------

    def print_report(self, professors: List[Dict[str, Any]], top_n: int = 10) -> None:
        """
        打印格式化的评分报告。

        Args:
            professors: 已评分的教授列表（需先调用 batch_score）
            top_n: 打印前N名
        """
        print("\n" + "=" * 70)
        print("  教授质量评分报告")
        print("=" * 70)
        print(f"  {'排名':<4} {'姓名':<25} {'总分':<5} {'研究':<5} {'产出':<5} {'职称':<5} {'地区':<5}")
        print("-" * 70)

        for i, prof in enumerate(professors[:top_n], 1):
            score = prof.get("_score", 0)
            bd = prof.get("_breakdown", {})
            name = prof.get("name", "?")[:24]
            passed = "✅" if prof.get("_passed") else "❌"

            print(
                f"  {passed} {i:<2} {name:<25} "
                f"{score:<5} {bd.get('research_match',0):<5} "
                f"{bd.get('publications',0):<5} {bd.get('rank',0):<5} "
                f"{bd.get('location',0):<5}"
            )

        print("-" * 70)
        if professors:
            avg = sum(p.get("_score", 0) for p in professors) / len(professors)
            passed = sum(1 for p in professors if p.get("_passed"))
            print(f"  平均分: {avg:.1f} | 通过阈值: {passed}/{len(professors)}")
        print("=" * 70 + "\n")

    def get_top_professors(
        self,
        professors: List[Dict[str, Any]],
        top_n: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        获取评分最高的 N 位教授（自动评分+排序+过滤）。

        Args:
            professors: 教授列表
            top_n: 返回前N名

        Returns:
            通过阈值的前N名教授
        """
        self.batch_score(professors, sort=True)
        # 只返回通过阈值的
        qualified = [p for p in professors if p.get("_passed", False)]
        return qualified[:top_n]


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

    print("=" * 70)
    print("  教授评分器 - 自测")
    print("=" * 70)

    # 加载档案
    parser = ProfileParser("profiles/my_profile_template.json")
    scorer = ProfessorScorer(parser)

    # 模拟教授数据
    mock_professors = [
        {
            "name": "Prof. Alpha AI",
            "institution": "MIT, United States",
            "research_topics": ["deep learning", "reinforcement learning", "ai for science"],
            "publication_count": 45,
            "h_index": 18,
            "citation_count": 3200,
            "recent_papers": [
                {"title": "Deep RL for Protein Folding", "year": 2025, "citations": 45, "venue": "Nature Machine Intelligence"},
                {"title": "Graph Neural Networks", "year": 2025, "citations": 30, "venue": "ICML"},
                {"title": "AI Drug Discovery", "year": 2024, "citations": 20, "venue": "NeurIPS"},
                {"title": "Molecular Dynamics", "year": 2024, "citations": 15, "venue": ""},
                {"title": "Protein Structure Prediction", "year": 2023, "citations": 60, "venue": "Nature"},
            ],
            "scholar_id": "001",
        },
        {
            "name": "Dr. Beta Biology",
            "institution": "University of Tokyo, Japan",
            "research_topics": ["medical imaging", "diagnosis", "cnn"],
            "publication_count": 12,
            "h_index": 8,
            "citation_count": 450,
            "recent_papers": [
                {"title": "Medical Image Analysis", "year": 2025, "citations": 8, "venue": ""},
                {"title": "Deep Diagnosis", "year": 2024, "citations": 5, "venue": ""},
            ],
            "scholar_id": "002",
        },
        {
            "name": "Prof. Gamma Science",
            "institution": "ETH Zurich, Switzerland",
            "research_topics": ["scientific discovery", "deep learning", "molecular dynamics", "protein structure prediction"],
            "publication_count": 80,
            "h_index": 35,
            "citation_count": 12000,
            "recent_papers": [
                {"title": "AI for Science", "year": 2026, "citations": 100, "venue": "Science"},
                {"title": "ML in Biology", "year": 2025, "citations": 80, "venue": "Nature Methods"},
                {"title": "Protein Design", "year": 2025, "citations": 55, "venue": "Nature Biotechnology"},
                {"title": "Molecular Simulation", "year": 2024, "citations": 40, "venue": ""},
                {"title": "Drug Discovery AI", "year": 2024, "citations": 35, "venue": ""},
            ],
            "scholar_id": "003",
        },
    ]

    # 评分
    print("\n[1] 批量评分...")
    scorer.batch_score(mock_professors)

    # 打印报告
    scorer.print_report(mock_professors)

    # 详细得分
    print("\n[2] 详细得分:")
    for prof in mock_professors:
        print(f"\n  {prof['name']}:")
        print(f"    总分: {prof['_score']}/100")
        print(f"    分项: 研究={prof['_breakdown']['research_match']} "
              f"产出={prof['_breakdown']['publications']} "
              f"职称={prof['_breakdown']['rank']} "
              f"地区={prof['_breakdown']['location']}")
        print(f"    匹配关键词: {prof['_details'].get('matched_keywords', [])}")
        print(f"    通过阈值: {prof['_passed']}")

    print("\n✅ 自测完成")
