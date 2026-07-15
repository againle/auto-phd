"""
论文批判性阅读器（Paper Reader）

生成高质量、可定制的论文摘要，包含：
- 核心贡献、方法论、关键结果
- 未解决问题（从论文原文提取）
- 与我研究的结合点（基于个人档案）
- 可探索的新方向 + 个人评估

支持：
- 全文阅读 vs 仅摘要模式
- 长文本智能截断（适配 LLM 上下文窗口）
- 缓存集成 + 断点续传
- 批量阅读教授论文
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from tqdm import tqdm

from scripts.llm_client import LLMClient, TaskType
from scripts.paper_cache import PaperCache
from scripts.utils import save_json, load_json, ensure_directory

logger = logging.getLogger(__name__)

# ============================================================
# Prompt 模板
# ============================================================

CRITICAL_READING_SYSTEM_PROMPT = """你是一位顶尖的学术评审专家，同时也是博士生申请顾问。
你的任务是深入阅读论文，生成批判性摘要，并帮助申请人找到与教授研究的具体结合点。

要求：
1. 必须具体、有建设性，禁止泛泛而谈
2. "未解决问题"必须引用论文中明确提到的局限性或未来工作
3. "结合点"必须基于申请人的实际技能和研究经历
4. "可探索的新方向"要有创新性但也要可行
5. 使用中文撰写（学术术语可保留英文）"""


CRITICAL_READING_PROMPT = """请阅读以下论文，并生成结构化批判性摘要。

## 我的学术背景
{profile_summary}

## 论文信息
- **标题**: {title}
- **作者**: {authors}
- **期刊/会议**: {journal}
- **年份**: {year}
- **引用数**: {citations}
- **摘要**: {abstract}
{full_text_section}

## 输出格式
请严格按照以下 Markdown 格式输出（每个部分都必须填写）：

---

# 论文批判性摘要

**论文标题**: {title}
**作者**: {authors}
**发表**: {journal}, {year}

## 核心贡献 (Core Contribution)
[用1-2句话总结本文的核心贡献，50-100字]

## 方法论 (Methodology)
[本文使用的主要技术/方法，100-200字，包括模型架构、训练策略、数据集等]

## 关键结果 (Key Results)
[主要实验结果和发现，100-150字，包含具体数字]

## 未解决的问题 (Unresolved Questions)
[作者明确提到的局限性或未来工作，必须从论文中提取，100-150字]

## 与我研究的结合点 (Connection to My Work)
[基于我的学术背景（见上文），提出2-3个具体的结合点：我可以如何利用我的技能延续或改进这项工作？150-200字]

## 可探索的新方向 (Novel Idea)
[基于本文，提出一个具有创新性但可行的新研究方向，100-150字]

## 个人评估 (Personal Assessment)
[对这篇论文的简短评价：创新性、实用性、对领域的重要性，50-100字]

---

请直接输出上述 Markdown，不要添加额外解释。"""


# ============================================================
# 文本截断工具
# ============================================================

def smart_truncate(text: str, max_chars: int = 6000, keep_ratio: float = 0.7) -> str:
    """
    智能截断长文本，优先保留开头和关键部分。

    策略：
    - 保留前 keep_ratio 的内容（通常包含引言和方法）
    - 从剩余部分提取关键句子（包含 "result", "experiment", "conclusion" 等）

    Args:
        text: 原始文本
        max_chars: 最大字符数
        keep_ratio: 保留头部比例

    Returns:
        截断后的文本
    """
    if len(text) <= max_chars:
        return text

    # 计算各部分大小
    head_size = int(max_chars * keep_ratio)
    tail_size = max_chars - head_size - 200  # 留 200 给分隔标记

    head = text[:head_size]

    # 从剩余部分提取关键段落
    remaining = text[head_size:]
    important_keywords = [
        "result", "experiment", "evaluation", "performance",
        "conclusion", "discussion", "limitation", "future work",
        "state-of-the-art", "compar", "ablation", "benchmark",
        "结果", "实验", "评估", "结论", "讨论",
    ]

    # 按段落分割
    paragraphs = remaining.split("\n\n")
    scored_paragraphs = []

    for para in paragraphs:
        para_lower = para.lower()
        score = sum(1 for kw in important_keywords if kw in para_lower)
        # 长段落加权
        score += min(len(para) / 500, 3)
        scored_paragraphs.append((score, para))

    # 选取得分最高的段落
    scored_paragraphs.sort(key=lambda x: x[0], reverse=True)
    selected = []
    current_len = 0
    for score, para in scored_paragraphs:
        if current_len + len(para) <= tail_size:
            selected.append(para)
            current_len += len(para)
        else:
            break

    tail = "\n\n".join(selected)

    return (
        f"{head}\n\n"
        f"[... 中间 {len(remaining)} 字符已省略，保留关键段落 ...]\n\n"
        f"{tail}"
    )


def estimate_tokens(text: str) -> int:
    """粗略估计 token 数（1 token ≈ 2 中文字符 ≈ 4 英文字符）"""
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    other_chars = len(text) - chinese_chars
    return chinese_chars // 2 + other_chars // 4


# ============================================================
# 论文阅读器
# ============================================================

class PaperReader:
    """
    论文批判性阅读器。

    使用示例:
        reader = PaperReader(profile_parser, llm_client, cache)
        result = reader.read_paper(paper_info, pdf_text)
        print(result["summary"])
    """

    def __init__(
        self,
        profile_parser,
        llm_client: LLMClient,
        cache: PaperCache,
        max_context_tokens: int = 6000,
    ):
        """
        Args:
            profile_parser: ProfileParser 实例
            llm_client: LLMClient 实例
            cache: PaperCache 实例
            max_context_tokens: LLM 上下文窗口上限（留给 prompt 的 token 数）
        """
        self.profile = profile_parser
        self.llm = llm_client
        self.cache = cache
        self.max_context_tokens = max_context_tokens

        # 我的档案摘要（缓存，避免重复生成）
        self._profile_short = self.profile.get_short_summary(max_length=300)
        self._profile_full = self.profile.get_profile_summary()

        # 统计
        self._read_stats = {
            "full_text": 0,
            "abstract_only": 0,
            "cached": 0,
            "failed": 0,
        }

        logger.info("论文阅读器初始化完成")

    # --------------------------------------------------------
    # 主入口
    # --------------------------------------------------------

    def read_paper(
        self,
        paper_info: Dict[str, Any],
        pdf_text: Optional[str] = None,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """
        阅读论文并生成批判性摘要。

        Args:
            paper_info: 论文信息（title, authors, abstract, year, journal, doi 等）
            pdf_text: PDF 全文文本（可选，如果已提取）
            force_refresh: 强制重新生成（忽略缓存）

        Returns:
            {
                "summary": "Markdown 摘要文本",
                "read_status": "success" | "failed",
                "mode": "full_text" | "abstract_only",
                "reason": "失败原因（如果失败）",
            }
        """
        title = paper_info.get("title", "Unknown")
        doi = paper_info.get("doi", "") or paper_info.get("paper_id", "") or title[:50]

        # 检查缓存
        if not force_refresh:
            cached = self.cache.get_summary(doi)
            if cached:
                self._read_stats["cached"] += 1
                logger.debug(f"摘要缓存命中: {title[:50]}")
                return {
                    "summary": cached,
                    "read_status": "success",
                    "mode": "cached",
                }

        # 确定使用全文还是仅摘要
        if pdf_text and len(pdf_text.strip()) > 200:
            mode = "full_text"
            self._read_stats["full_text"] += 1
            logger.info(f"全文阅读: {title[:60]}")
        else:
            mode = "abstract_only"
            self._read_stats["abstract_only"] += 1
            logger.info(f"摘要阅读: {title[:60]}")

        try:
            # 构建 prompt
            prompt = self._build_prompt(paper_info, pdf_text)

            # 调用 LLM
            response = self.llm.call(
                messages=[
                    {"role": "system", "content": CRITICAL_READING_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                task_type=TaskType.PAPER_SUMMARY,
                temperature=0.5,  # 降低温度以获得更一致的输出
            )

            # 缓存摘要
            self.cache.cache_summary(
                doi,
                response,
                metadata={
                    "title": title,
                    "mode": mode,
                    "generated_at": datetime.now().isoformat(),
                },
            )

            logger.info(f"✅ 论文阅读完成 [{mode}]: {title[:60]}")

            return {
                "summary": response,
                "read_status": "success",
                "mode": mode,
            }

        except Exception as e:
            self._read_stats["failed"] += 1
            logger.error(f"论文阅读失败 '{title[:60]}': {e}")
            return {
                "summary": "",
                "read_status": "failed",
                "mode": mode,
                "reason": str(e)[:200],
            }

    # --------------------------------------------------------
    # Prompt 构建
    # --------------------------------------------------------

    def _build_prompt(
        self,
        paper_info: Dict[str, Any],
        pdf_text: Optional[str] = None,
    ) -> str:
        """
        构建摘要生成 Prompt。

        包含：我的档案 + 论文信息 + 全文（可选）
        """
        title = paper_info.get("title", "Unknown")
        authors_raw = paper_info.get("authors", [])
        if isinstance(authors_raw, list):
            authors = ", ".join(
                a.get("name", a) if isinstance(a, dict) else str(a)
                for a in authors_raw[:8]
            )
        else:
            authors = str(authors_raw)

        journal_raw = paper_info.get("journal", "") or paper_info.get("venue", "")
        if isinstance(journal_raw, dict):
            journal = journal_raw.get("name", journal_raw.get("journal", ""))
        elif isinstance(journal_raw, str):
            journal = journal_raw
        else:
            journal = ""

        year = paper_info.get("year", "")
        abstract = paper_info.get("abstract", "") or "（摘要不可用）"
        citations = paper_info.get("citations", paper_info.get("citation_count", 0))

        # 处理全文
        full_text_section = ""
        if pdf_text and len(pdf_text.strip()) > 200:
            # 估计 token 数，智能截断
            profile_tokens = estimate_tokens(self._profile_full)
            prompt_tokens = estimate_tokens(
                CRITICAL_READING_PROMPT.format(
                    profile_summary="",
                    title="",
                    authors="",
                    journal="",
                    year="",
                    citations="",
                    abstract="",
                    full_text_section="",
                )
            )
            available_tokens = self.max_context_tokens - profile_tokens - prompt_tokens - 800
            max_chars = max(available_tokens * 4, 2000)  # 至少 2000 字符

            truncated = smart_truncate(pdf_text, max_chars=int(max_chars))
            full_text_section = f"\n- **全文**（已截取关键部分）:\n```\n{truncated}\n```"

        return CRITICAL_READING_PROMPT.format(
            profile_summary=self._profile_full,
            title=title,
            authors=authors,
            journal=journal or "Unknown",
            year=year or "N/A",
            citations=citations,
            abstract=abstract[:1500],  # 截断过长摘要
            full_text_section=full_text_section,
        )

    # --------------------------------------------------------
    # 批量阅读
    # --------------------------------------------------------

    def batch_read(
        self,
        professor: Dict[str, Any],
        max_papers: int = 3,
        pdf_texts: Optional[Dict[str, str]] = None,
        prof_dir: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        批量阅读教授的论文。

        Args:
            professor: 教授信息字典
            max_papers: 最多阅读几篇
            pdf_texts: {paper_title: extracted_text} 预提取的 PDF 文本
            prof_dir: 教授文件夹路径

        Returns:
            阅读结果列表
        """
        papers = professor.get("recent_papers", [])
        name = professor.get("name", "Unknown")

        # 按引用数排序
        sorted_papers = sorted(
            papers,
            key=lambda p: p.get("citations", 0),
            reverse=True,
        )[:max_papers]

        if not sorted_papers:
            logger.warning(f"教授 {name} 无可用论文")
            return []

        logger.info(f"开始批量阅读 {name} 的 {len(sorted_papers)} 篇论文")
        results = []

        for paper in tqdm(sorted_papers, desc=f"阅读 {name[:20]}", unit="paper", ncols=80):
            title = paper.get("title", "Unknown")

            # 获取 PDF 文本（如果提供）
            pdf_text = None
            if pdf_texts:
                pdf_text = pdf_texts.get(title, pdf_texts.get(title[:50], None))

            # 阅读
            result = self.read_paper(paper, pdf_text=pdf_text)
            result["professor_name"] = name
            result["paper_title"] = title
            results.append(result)

            # 保存到教授文件夹
            if prof_dir and result["read_status"] == "success":
                self._save_to_professor_folder(
                    prof_dir, paper, result["summary"]
                )

        # 统计
        success = sum(1 for r in results if r["read_status"] == "success")
        logger.info(
            f"{name} 论文阅读完成: {success}/{len(results)} 成功 "
            f"(full_text: {self._read_stats['full_text']}, "
            f"abstract_only: {self._read_stats['abstract_only']}, "
            f"cached: {self._read_stats['cached']})"
        )

        return results

    # --------------------------------------------------------
    # 保存
    # --------------------------------------------------------

    def _save_to_professor_folder(
        self,
        prof_dir: str,
        paper_info: Dict,
        summary: str,
    ) -> None:
        """将摘要保存到教授文件夹下的 papers/ 目录"""
        papers_dir = Path(prof_dir) / "papers"
        ensure_directory(str(papers_dir))

        title = paper_info.get("title", "unknown")[:60]
        safe_title = re.sub(r'[<>:"/\\|?*]', '_', title)
        summary_path = papers_dir / f"{safe_title}_summary.md"

        summary_path.write_text(summary, encoding="utf-8")
        logger.debug(f"摘要已保存: {summary_path}")

    # --------------------------------------------------------
    # 统计
    # --------------------------------------------------------

    def get_read_stats(self) -> Dict[str, int]:
        """返回阅读统计"""
        return dict(self._read_stats)

    def reset_stats(self) -> None:
        """重置统计"""
        for key in self._read_stats:
            self._read_stats[key] = 0


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
    print("  论文阅读器 - 自测")
    print(f"{'═' * 60}")

    parser = ProfileParser("profiles/my_profile_template.json")
    client = LLMClient()
    cache = PaperCache()
    reader = PaperReader(parser, client, cache)

    # 测试1: 仅摘要模式
    print("\n[1] 仅摘要模式阅读...")
    paper_info = {
        "title": "Deep Learning for Protein Structure Prediction",
        "authors": [{"name": "John Jumper"}, {"name": "Richard Evans"}, {"name": "Alexander Pritzel"}],
        "journal": "Nature",
        "year": 2021,
        "citations": 15000,
        "abstract": (
            "Proteins are essential to life, and understanding their structure can "
            "facilitate a mechanistic understanding of their function. Through an enormous "
            "experimental effort, the structures of around 100,000 unique proteins have "
            "been determined, but this represents a small fraction of the billions of "
            "known protein sequences. Here we present AlphaFold, a neural network-based "
            "approach to predicting protein structures from amino acid sequences. "
            "We demonstrate accuracy competitive with experimental structures in "
            "the majority of cases."
        ),
        "paper_id": "test_alphafold",
    }

    result = reader.read_paper(paper_info)
    if result["read_status"] == "success":
        print(f"    ✅ 模式: {result['mode']}")
        print(f"    摘要前200字:")
        print(f"    {result['summary'][:200]}...")
    else:
        print(f"    ❌ 失败: {result.get('reason', 'Unknown')}")

    # 测试2: 全文模式
    print("\n[2] 全文模式阅读...")
    mock_full_text = """
    Abstract
    We propose a novel deep learning architecture for protein structure prediction.
    
    1. Introduction
    Protein structure prediction is a fundamental problem in computational biology...
    
    2. Method
    Our approach uses a transformer-based architecture with attention mechanisms...
    The model is trained on the Protein Data Bank (PDB) dataset with 170,000 structures.
    
    3. Experiments
    We evaluate on CASP14 benchmark and achieve state-of-the-art results with
    TM-score of 0.92, improving over previous methods by 15%.
    
    4. Results
    Our method outperforms all existing approaches on the CASP14 benchmark.
    The average RMSD is 1.2 Angstroms, compared to 2.5 for the previous best method.
    
    5. Limitations
    Our method requires significant computational resources (500 GPU-days).
    It does not handle multi-chain protein complexes well.
    Future work should address the problem of protein dynamics and folding pathways.
    
    6. Conclusion
    We present a significant advance in protein structure prediction...
    """

    result2 = reader.read_paper(paper_info, pdf_text=mock_full_text, force_refresh=True)
    if result2["read_status"] == "success":
        print(f"    ✅ 模式: {result2['mode']}")
        # 检查各节是否齐全
        required_sections = [
            "核心贡献", "方法论", "关键结果",
            "未解决问题", "结合点", "新方向", "个人评估",
        ]
        found = [s for s in required_sections if s in result2["summary"]]
        print(f"    包含章节: {len(found)}/7 ({', '.join(found)})")
    else:
        print(f"    ❌ 失败: {result2.get('reason', 'Unknown')}")

    # 测试3: 批量阅读
    print("\n[3] 批量阅读测试...")
    mock_professor = {
        "name": "Test Professor",
        "recent_papers": [paper_info],
    }
    results = reader.batch_read(mock_professor, max_papers=1)
    print(f"    批量结果: {len(results)} 篇, 成功 {sum(1 for r in results if r['read_status'] == 'success')}")

    # 统计
    print(f"\n[4] 阅读统计:")
    for k, v in reader.get_read_stats().items():
        print(f"    {k}: {v}")

    cache.print_stats()

    print(f"\n✅ 自测完成")
