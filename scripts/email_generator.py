"""
个性化套磁信生成器（Email Generator）

功能：
- 读取教授信息 + 论文批判性摘要
- 智能选择最适合引用的论文
- 调用 DeepSeek 生成 2-3 个版本的套磁信
- 保存草稿到教授文件夹（支持手动编辑）
- 生成日志追踪

输出草稿：
  professors/{name}/drafts/
  ├── v1_academic.md
  ├── v2_concise.md
  └── v3_enthusiastic.md
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from tqdm import tqdm

from scripts.llm_client import LLMClient, TaskType
from scripts.email_templates import EmailTemplateManager
from scripts.utils import load_json, save_json, ensure_directory

logger = logging.getLogger(__name__)

# ============================================================
# Prompt 模板
# ============================================================

EMAIL_SYSTEM_PROMPT = """You are an expert academic email writer helping a PhD student craft personalized cold emails to professors. 

Key principles:
1. Be SPECIFIC — reference the professor's actual paper and show you've read it
2. Be CONCISE — professors are busy, 150-250 words is ideal
3. Be GENUINE — show real intellectual curiosity, not flattery
4. Connect YOUR skills to THEIR research challenges
5. NEVER use generic templates or vague praise"""


EMAIL_GENERATION_PROMPT = """Write a personalized academic cold email for a PhD application.

## My Profile
Name: {my_name}
Current: {my_position} at {my_institution}
Research interests: {my_interests}
Key skills: {my_skills}
Recent publication: {my_publication}

## Target Professor
Name: {professor_name}
Institution: {professor_institution}
Research topics: {professor_topics}

## Paper I've Read (to reference)
Title: {paper_title}
Core contribution: {core_contribution}
Open question raised: {unresolved_question}
Connection to my work: {my_connection}

## Style Guidelines
Style: {style_name} ({style_tone} tone)
Opening approach: {style_opening}

## Requirements
1. Paragraph 1: Reference the professor's specific paper — show you read and understood it. Mention the core contribution.
2. Paragraph 2: Connect the paper's open question to YOUR research background. Be specific about what skills you bring.
3. Paragraph 3: Briefly close with your interest in joining their group and a soft ask (meeting, application, etc.)
4. Keep it 150-250 words. Professional, warm, not desperate.
5. English only.

## Output Format
Subject: [email subject line]

[email body — no markdown formatting needed, just plain paragraphs]"""


# ============================================================
# 论文选择 Prompt
# ============================================================

PAPER_SELECTION_PROMPT = """You are helping a PhD student choose the best paper to reference when emailing a professor.

## My Research Interests
{my_interests}

## Papers from this Professor
{paper_list}

## Task
Which paper would create the strongest connection between my background and the professor's work?
Consider: (1) overlap with my interests, (2) recency, (3) open questions I could address.

Return ONLY a JSON object:
{{"selected_index": 0, "reason": "Brief reason (one sentence)"}}
"""


# ============================================================
# 邮件生成器
# ============================================================

class EmailGenerator:
    """
    个性化套磁信生成器。

    使用示例:
        generator = EmailGenerator(profile_parser, llm_client)
        result = generator.generate_emails(Path("professors/MIT_Alice/"))
        # → 3 个版本保存到 drafts/ 目录
    """

    def __init__(
        self,
        profile_parser,
        llm_client: LLMClient,
        template_mgr: Optional[EmailTemplateManager] = None,
    ):
        """
        Args:
            profile_parser: ProfileParser 实例
            llm_client: LLMClient 实例
            template_mgr: EmailTemplateManager 实例（可选）
        """
        self.profile = profile_parser
        self.llm = llm_client
        self.templates = template_mgr or EmailTemplateManager()

        # 缓存我的信息
        self._my_name = self.profile.profile.personal_info.name
        self._my_email = self.profile.profile.personal_info.email
        self._my_position = self.profile.profile.personal_info.current_position
        self._my_institution = self.profile.profile.personal_info.current_institution

        # 风格配置
        self.style_configs = {
            "academic": {
                "name": "Academic Formal",
                "tone": "formal and professional",
                "opening": "Express strong interest in their specific research, citing the paper title",
            },
            "concise": {
                "name": "Direct & Concise",
                "tone": "direct and efficient",
                "opening": "Get straight to the point — who you are, what paper you read, why you're writing",
            },
            "enthusiastic": {
                "name": "Enthusiastic & Warm",
                "tone": "warm and enthusiastic",
                "opening": "Show genuine excitement about their research direction and your desire to contribute",
            },
        }

        # 统计
        self._stats = {"generated": 0, "failed": 0, "cached": 0}

        logger.info("邮件生成器初始化完成")

    # --------------------------------------------------------
    # 主入口
    # --------------------------------------------------------

    def generate_emails(
        self,
        professor_folder: Path,
        styles: Optional[List[str]] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        为教授生成多版本套磁信。

        Args:
            professor_folder: 教授文件夹路径
            styles: 要生成的风格列表，默认 ["academic", "concise", "enthusiastic"]
            force: 是否强制重新生成

        Returns:
            {
                "versions": {style: {subject, body, style_label}},
                "best_paper": {title, doi, reason},
                "quality_score": int,
                "recommended_version": str,
                "professor_name": str,
            }
        """
        if styles is None:
            styles = ["academic", "concise", "enthusiastic"]

        prof_dir = Path(professor_folder)
        info_path = prof_dir / "info.json"

        if not info_path.exists():
            raise FileNotFoundError(f"info.json 不存在: {info_path}")

        # 加载教授信息
        prof_info = load_json(str(info_path))
        name = prof_info.get("name", prof_dir.name)
        logger.info(f"开始为 {name} 生成套磁信")

        # 选择最佳论文
        best_paper = self._select_best_paper(prof_dir, prof_info)
        if not best_paper:
            logger.warning(f"无法为 {name} 选择论文，使用第一页论文")
            papers = prof_info.get("recent_papers", [])
            if not papers:
                return {
                    "versions": {},
                    "best_paper": None,
                    "quality_score": 0,
                    "recommended_version": "",
                    "professor_name": name,
                    "error": "无可用论文",
                }
            best_paper = {
                "title": papers[0].get("title", ""),
                "paper_id": papers[0].get("paper_id", ""),
                "reason": "默认选择（无更多信息）",
            }

        # 加载该论文的摘要
        summary = self._load_paper_summary(prof_dir, best_paper)

        # 检查草稿是否已存在
        drafts_dir = prof_dir / "drafts"
        existing = self._find_existing_drafts(drafts_dir, styles)

        versions = {}
        for style in styles:
            if style in existing and not force:
                # 使用已有草稿
                draft_content = (drafts_dir / existing[style]).read_text(encoding="utf-8")
                parsed = self._parse_email(draft_content)
                versions[style] = {
                    "subject": parsed["subject"],
                    "body": parsed["body"],
                    "style_label": self.style_configs.get(style, {}).get("name", style),
                    "source": "cached",
                }
                self._stats["cached"] += 1
            else:
                # 生成新邮件
                try:
                    email_data = self._generate_single(
                        prof_info, best_paper, summary, style
                    )
                    versions[style] = email_data
                    self._stats["generated"] += 1
                except Exception as e:
                    logger.error(f"生成 {style} 版本失败: {e}")
                    self._stats["failed"] += 1
                    versions[style] = {
                        "subject": f"[生成失败] {best_paper.get('title', '')}",
                        "body": f"生成失败: {str(e)[:200]}",
                        "style_label": style,
                        "source": "failed",
                    }

        # 保存草稿
        self.save_drafts(prof_dir, versions, best_paper)

        # 推荐最佳版本
        recommended = self._recommend_version(versions)

        # 计算质量分
        quality_score = self._calculate_quality(prof_info, versions)

        result = {
            "versions": versions,
            "best_paper": {
                "title": best_paper.get("title", ""),
                "paper_id": best_paper.get("paper_id", ""),
                "reason": best_paper.get("reason", ""),
            },
            "quality_score": quality_score,
            "recommended_version": recommended,
            "professor_name": name,
        }

        logger.info(
            f"{name} 邮件生成完成: {len(versions)} 版本, "
            f"推荐 {recommended}, 质量分 {quality_score}"
        )

        return result

    # --------------------------------------------------------
    # 论文选择
    # --------------------------------------------------------

    def _select_best_paper(
        self, prof_dir: Path, prof_info: Dict
    ) -> Optional[Dict[str, str]]:
        """
        选择最适合引用的论文。

        策略：
        1. 优先选择有完整摘要的论文
        2. 考虑引用数（高引用 = 重要）
        3. 考虑与我的研究关键词重叠度
        """
        papers = prof_info.get("recent_papers", [])
        if not papers:
            return None

        # 查找已有的摘要文件
        papers_dir = prof_dir / "papers"
        papers_with_summaries = []
        papers_without = []

        for paper in papers:
            title = paper.get("title", "")
            safe_title = re.sub(r'[<>:"/\\|?*]', '_', title)
            summary_path = papers_dir / f"{safe_title}_summary.md"

            if summary_path.exists():
                papers_with_summaries.append((paper, summary_path))
            else:
                papers_without.append(paper)

        # 优先使用有摘要的论文
        candidates = papers_with_summaries if papers_with_summaries else [
            (p, None) for p in papers_without
        ]

        if not candidates:
            return None

        if len(candidates) == 1:
            paper, _ = candidates[0]
            return {
                "title": paper.get("title", ""),
                "paper_id": paper.get("paper_id", ""),
                "reason": "唯一候选论文",
            }

        # 多篇候选时：使用 LLM 选择
        my_keywords = ", ".join(self.profile.get_research_keywords()[:8])

        paper_list = []
        for i, (paper, _) in enumerate(candidates[:5]):
            paper_list.append(
                f"{i}. \"{paper.get('title', 'Unknown')}\" "
                f"({paper.get('year', '?')}, {paper.get('citations', 0)} citations)"
            )

        prompt = PAPER_SELECTION_PROMPT.format(
            my_interests=my_keywords,
            paper_list="\n".join(paper_list),
        )

        try:
            response = self.llm.call(
                messages=[{"role": "user", "content": prompt}],
                task_type=TaskType.GENERAL,
            )
            # 解析 JSON
            cleaned = response.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                cleaned = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else cleaned
            choice = json.loads(cleaned)
            idx = choice.get("selected_index", 0)
            reason = choice.get("reason", "LLM 推荐")
        except Exception as e:
            logger.warning(f"LLM 论文选择失败，使用启发式: {e}")
            # 回退：选引用数最高的
            idx = 0
            reason = "引用数最高（回退）"

        # 确保索引有效
        idx = max(0, min(idx, len(candidates) - 1))
        paper, _ = candidates[idx]

        return {
            "title": paper.get("title", ""),
            "paper_id": paper.get("paper_id", ""),
            "doi": paper.get("doi", ""),
            "reason": reason,
        }

    # --------------------------------------------------------
    # 单封邮件生成
    # --------------------------------------------------------

    def _generate_single(
        self,
        prof_info: Dict,
        best_paper: Dict,
        summary: Dict[str, str],
        style: str,
    ) -> Dict[str, str]:
        """生成单个风格的套磁信"""

        style_conf = self.style_configs.get(style, self.style_configs["academic"])

        # 我的代表作
        publications = self.profile.profile.publications
        if publications:
            my_pub = publications[0]
            my_pub_str = (
                f"\"{my_pub.title}\" ({my_pub.journal}, {my_pub.year})"
                f" — {my_pub.contribution_summary}"
            )
        else:
            my_pub_str = "（待填写）"

        prompt = EMAIL_GENERATION_PROMPT.format(
            my_name=self._my_name,
            my_position=self._my_position,
            my_institution=self._my_institution,
            my_interests=", ".join(self.profile.get_research_keywords()[:8]),
            my_skills=", ".join(self.profile.get_skills()[:8]),
            my_publication=my_pub_str,
            professor_name=prof_info.get("name", ""),
            professor_institution=prof_info.get("institution", ""),
            professor_topics=", ".join(prof_info.get("research_topics", [])[:5]),
            paper_title=best_paper.get("title", ""),
            core_contribution=summary.get("core_contribution", best_paper.get("reason", "")),
            unresolved_question=summary.get("unresolved_question", ""),
            my_connection=summary.get("connection_to_my_work", summary.get("my_connection", "")),
            style_name=style_conf["name"],
            style_tone=style_conf["tone"],
            style_opening=style_conf["opening"],
        )

        response = self.llm.call(
            messages=[
                {"role": "system", "content": EMAIL_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            task_type=TaskType.EMAIL_GENERATION,
            temperature=0.7,
        )

        parsed = self._parse_email(response)

        return {
            "subject": parsed["subject"],
            "body": parsed["body"],
            "style_label": style_conf["name"],
            "source": "generated",
            "generated_at": datetime.now().isoformat(),
        }

    # --------------------------------------------------------
    # 摘要加载
    # --------------------------------------------------------

    def _load_paper_summary(
        self, prof_dir: Path, paper: Dict
    ) -> Dict[str, str]:
        """加载论文的批判性摘要"""
        title = paper.get("title", "")
        safe_title = re.sub(r'[<>:"/\\|?*]', '_', title)
        summary_path = prof_dir / "papers" / f"{safe_title}_summary.md"

        if not summary_path.exists():
            logger.debug(f"摘要文件不存在: {summary_path}")
            return {}

        try:
            content = summary_path.read_text(encoding="utf-8")

            # 解析各节
            sections = {}
            current_section = "body"
            for line in content.split("\n"):
                line = line.strip()
                # 匹配 ## 标题
                match = re.match(r'^##\s+(.+?)(?:\s*\(.+?\))?\s*$', line)
                if match:
                    section_name = match.group(1).strip().lower()
                    # 映射到标准键名
                    key_map = {
                        "核心贡献": "core_contribution",
                        "core contribution": "core_contribution",
                        "方法论": "methodology",
                        "methodology": "methodology",
                        "关键结果": "key_results",
                        "key results": "key_results",
                        "未解决问题": "unresolved_question",
                        "未解决的问题": "unresolved_question",
                        "unresolved questions": "unresolved_question",
                        "与我研究的结合点": "connection_to_my_work",
                        "connection to my work": "connection_to_my_work",
                        "可探索的新方向": "novel_idea",
                        "novel idea": "novel_idea",
                        "个人评估": "personal_assessment",
                        "personal assessment": "personal_assessment",
                    }
                    current_section = key_map.get(section_name, section_name)
                    sections[current_section] = ""
                elif current_section and line:
                    sections[current_section] = (
                        sections.get(current_section, "") + line + " "
                    )

            # 清理多余空格
            return {k: v.strip() for k, v in sections.items() if v.strip()}

        except Exception as e:
            logger.warning(f"解析摘要失败 {summary_path}: {e}")
            return {}

    # --------------------------------------------------------
    # 解析与保存
    # --------------------------------------------------------

    def _parse_email(self, text: str) -> Dict[str, str]:
        """从 LLM 响应中解析 Subject 和 Body"""
        text = text.strip()

        subject = ""
        body = text

        # 匹配 "Subject: ..." 模式
        subject_match = re.match(
            r'(?:^|\n)\s*Subject:\s*(.+?)(?:\n|$)',
            text,
            re.IGNORECASE,
        )
        if subject_match:
            subject = subject_match.group(1).strip()
            # 去掉 subject 行，剩余为 body
            body = text[subject_match.end():].strip()

        # 去掉可能的 markdown 标记
        body = re.sub(r'^---+$', '', body, flags=re.MULTILINE).strip()

        if not subject:
            # 尝试取第一行作为 subject
            lines = body.split("\n")
            if lines:
                first_line = lines[0].strip()
                if len(first_line) < 120 and not first_line.startswith("Dear"):
                    subject = first_line
                    body = "\n".join(lines[1:]).strip()

        return {
            "subject": subject or "[No Subject]",
            "body": body,
        }

    def save_drafts(
        self,
        prof_dir: Path,
        versions: Dict[str, Dict],
        best_paper: Dict,
    ) -> None:
        """
        保存所有版本的草稿到 drafts/ 目录。

        文件格式:
        - drafts/v1_academic.md
        - drafts/v2_concise.md
        - drafts/metadata.json
        """
        drafts_dir = prof_dir / "drafts"
        ensure_directory(str(drafts_dir))

        for i, (style, data) in enumerate(versions.items(), 1):
            if data.get("source") == "failed":
                continue

            filename = f"v{i}_{style}.md"
            filepath = drafts_dir / filename

            content = (
                f"# 套磁信草稿 — {style.upper()}\n\n"
                f"**生成时间**: {data.get('generated_at', datetime.now().isoformat())}\n"
                f"**风格**: {data.get('style_label', style)}\n"
                f"**引用论文**: {best_paper.get('title', '')}\n"
                f"**选择理由**: {best_paper.get('reason', '')}\n\n"
                f"---\n\n"
                f"**Subject**: {data['subject']}\n\n"
                f"{data['body']}\n\n"
                f"---\n\n"
                f"*生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')} "
                f"| 请检查并修改后再发送*"
            )

            filepath.write_text(content, encoding="utf-8")
            logger.debug(f"草稿已保存: {filepath}")

        # 保存元数据
        metadata = {
            "generated_at": datetime.now().isoformat(),
            "best_paper": best_paper,
            "versions": list(versions.keys()),
        }
        save_json(metadata, str(drafts_dir / "metadata.json"))

    # --------------------------------------------------------
    # 辅助
    # --------------------------------------------------------

    def _find_existing_drafts(
        self, drafts_dir: Path, styles: List[str]
    ) -> Dict[str, str]:
        """查找已存在的草稿"""
        existing = {}
        if not drafts_dir.exists():
            return existing

        for f in drafts_dir.glob("v*_*.md"):
            for style in styles:
                if f.name.endswith(f"_{style}.md"):
                    existing[style] = f.name
        return existing

    def _recommend_version(self, versions: Dict) -> str:
        """推荐最佳版本（默认推荐 academic）"""
        # 简单策略：优先 academic，否则第一个
        if "academic" in versions and versions["academic"].get("source") != "failed":
            return "academic"
        for style, data in versions.items():
            if data.get("source") != "failed":
                return style
        return list(versions.keys())[0] if versions else ""

    def _calculate_quality(
        self, prof_info: Dict, versions: Dict
    ) -> int:
        """计算邮件质量分（0-100）"""
        score = 60  # 基础分

        # 有多个版本
        score += min(len(versions) * 5, 15)

        # 教授信息完整
        if prof_info.get("h_index", 0) > 0:
            score += 5
        if prof_info.get("research_topics"):
            score += 5

        # 检查邮件长度是否合理（150-250 words）
        for data in versions.values():
            body = data.get("body", "")
            word_count = len(body.split())
            if 120 <= word_count <= 280:
                score += 5
                break

        return min(score, 100)

    # --------------------------------------------------------
    # 批量生成
    # --------------------------------------------------------

    def batch_generate(
        self,
        professor_folders: List[Path],
        styles: Optional[List[str]] = None,
        force: bool = False,
    ) -> List[Dict]:
        """
        批量为多位教授生成套磁信。

        Returns:
            结果列表
        """
        if styles is None:
            styles = ["academic", "concise"]

        results = []
        for prof_dir in tqdm(professor_folders, desc="生成邮件", unit="prof", ncols=80):
            try:
                result = self.generate_emails(prof_dir, styles=styles, force=force)
                results.append(result)
            except Exception as e:
                logger.error(f"{prof_dir.name} 邮件生成失败: {e}")
                results.append({
                    "versions": {},
                    "best_paper": None,
                    "professor_name": prof_dir.name,
                    "error": str(e),
                })

        return results

    def get_stats(self) -> Dict[str, int]:
        return dict(self._stats)


# ============================================================
# 自测入口
# ============================================================

if __name__ == "__main__":
    from scripts.profile_parser import ProfileParser
    from scripts.paper_cache import PaperCache

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    print(f"\n{'═' * 60}")
    print("  套磁信生成器 - 自测")
    print(f"{'═' * 60}")

    parser = ProfileParser("profiles/my_profile_template.json")
    client = LLMClient()
    generator = EmailGenerator(parser, client)

    # 查找有论文摘要的教授文件夹
    prof_dirs = sorted(Path("professors").glob("*"))
    valid_dirs = [
        d for d in prof_dirs
        if d.is_dir()
        and (d / "info.json").exists()
        and list((d / "papers").glob("*_summary.md"))
    ]

    if valid_dirs:
        print(f"\n找到 {len(valid_dirs)} 个有摘要的教授文件夹")

        # 只生成 1 个版本节省 API 调用
        print(f"\n[1] 为 {valid_dirs[0].name} 生成邮件 (academic only)...")
        result = generator.generate_emails(valid_dirs[0], styles=["academic"])

        if result.get("versions"):
            v = result["versions"].get("academic", {})
            print(f"\n  最佳论文: {result['best_paper'].get('title', '')[:60]}")
            print(f"  选择理由: {result['best_paper'].get('reason', '')}")
            print(f"  质量分: {result['quality_score']}/100")
            print(f"  推荐版本: {result['recommended_version']}")
            print(f"\n  Subject: {v.get('subject', '')}")
            print(f"\n  Body (前200字):")
            print(f"  {v.get('body', '')[:200]}...")

            # 显示草稿文件
            print(f"\n[2] 草稿文件:")
            drafts_dir = valid_dirs[0] / "drafts"
            for f in sorted(drafts_dir.glob("*")):
                print(f"    {f.name}")
    else:
        print("\n⚠️ 没有找到带摘要的教授文件夹。")
        print("   请先运行 process_professor 完成论文阅读。")

        # 创建测试场景
        print("\n[测试] 创建模拟数据...")
        from scripts.utils import save_json
        test_dir = Path("professors/Test_Email_Prof")
        ensure_directory(str(test_dir))
        ensure_directory(str(test_dir / "papers"))
        ensure_directory(str(test_dir / "drafts"))

        # 测试 info.json
        save_json({
            "name": "Dr. Email Test",
            "institution": "Stanford University",
            "research_topics": ["deep learning", "protein structure", "AI for Science"],
            "publication_count": 25,
            "h_index": 15,
            "recent_papers": [{
                "title": "Deep Learning Revolution in Protein Folding",
                "year": 2025, "citations": 150, "paper_id": "test_p1",
            }],
            "status": "paper_reading_completed",
        }, str(test_dir / "info.json"))

        # 测试摘要
        summary_md = """## 核心贡献 (Core Contribution)
Proposed a novel Transformer architecture for protein structure prediction.

## 未解决的问题 (Unresolved Questions)
The method struggles with multi-chain protein complexes and requires 500 GPU-days.

## 与我研究的结合点 (Connection to My Work)
I have experience with graph neural networks for protein interactions."""
        (test_dir / "papers" / "Deep Learning Revolution in Protein Folding_summary.md").write_text(summary_md)

        print("    ✅ 测试数据已创建")
        result = generator.generate_emails(test_dir, styles=["academic"])

        if result.get("versions"):
            v = result["versions"]["academic"]
            print(f"\n   Subject: {v['subject']}")
            print(f"   Body preview: {v['body'][:150]}...")

    print(f"\n[统计] 生成: {generator.get_stats()['generated']}, "
          f"缓存: {generator.get_stats()['cached']}")

    print(f"\n✅ 自测完成")
