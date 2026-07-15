"""
论文缓存管理器（Paper Cache）

功能：
- 缓存 PDF 论文到 papers_cache/ 目录
- 缓存 LLM 生成的摘要到 papers_cache/summaries/
- 避免重复下载和重复调用 LLM
- 缓存元数据追踪 + 统计
- 线程安全

文件命名规则：
  PDF:   papers_cache/{doi_safe}.pdf
  摘要:  papers_cache/summaries/{doi_safe}_summary.md
  元数据: papers_cache/cache_metadata.json
"""

import hashlib
import json
import logging
import re
import shutil
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Any, List

from scripts.utils import ensure_directory, load_json, save_json

logger = logging.getLogger(__name__)

# ============================================================
# 常量
# ============================================================

METADATA_FILE = "papers_cache/cache_metadata.json"


# ============================================================
# 工具函数
# ============================================================

def _doi_to_safe_filename(identifier: str) -> str:
    """
    将 DOI 或论文 ID 转换为安全的文件名。

    Args:
        identifier: DOI (如 "10.1038/s42256-025-01234-5") 或 paper_id

    Returns:
        安全文件名（不含扩展名），如 "10.1038_s42256-025-01234-5"
    """
    # 替换路径分隔符和特殊字符
    safe = identifier.replace("/", "_").replace("\\", "_").replace(":", "_")
    safe = re.sub(r'[<>"|?*]', "", safe)
    safe = safe.strip("_ .")
    if not safe:
        safe = hashlib.md5(identifier.encode()).hexdigest()[:12]
    return safe


# ============================================================
# 缓存元数据管理
# ============================================================

class CacheMetadata:
    """缓存元数据管理器（线程安全）"""

    def __init__(self, metadata_path: str = METADATA_FILE):
        self._path = Path(metadata_path)
        self._lock = threading.Lock()
        self._data: Dict[str, Dict] = {}
        self._load()

    def _load(self) -> None:
        """加载元数据"""
        if self._path.exists():
            try:
                self._data = load_json(str(self._path))
            except Exception as e:
                logger.warning(f"加载缓存元数据失败: {e}")
                self._data = {}

    def _save(self) -> None:
        """保存元数据"""
        try:
            ensure_directory(str(self._path.parent))
            with self._lock:
                save_json(self._data, str(self._path))
        except Exception as e:
            logger.warning(f"保存缓存元数据失败: {e}")

    def has(self, doi_safe: str, entry_type: str = "pdf") -> bool:
        """检查是否有某条缓存记录"""
        with self._lock:
            key = f"{doi_safe}_{entry_type}"
            return key in self._data

    def get(self, doi_safe: str, entry_type: str = "pdf") -> Optional[Dict]:
        """获取缓存记录"""
        with self._lock:
            key = f"{doi_safe}_{entry_type}"
            return self._data.get(key)

    def set(
        self,
        doi_safe: str,
        entry_type: str,
        file_path: str,
        file_size: int = 0,
        extra: Optional[Dict] = None,
    ) -> None:
        """设置缓存记录"""
        with self._lock:
            key = f"{doi_safe}_{entry_type}"
            self._data[key] = {
                "identifier": doi_safe,
                "type": entry_type,
                "file_path": file_path,
                "file_size_bytes": file_size,
                "cached_at": datetime.now().isoformat(),
            }
            if extra:
                self._data[key].update(extra)
        self._save()

    def remove(self, doi_safe: str, entry_type: str = "pdf") -> None:
        """删除缓存记录"""
        with self._lock:
            key = f"{doi_safe}_{entry_type}"
            self._data.pop(key, None)
        self._save()

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        with self._lock:
            pdf_count = 0
            summary_count = 0
            total_size = 0
            oldest = None
            newest = None

            for entry in self._data.values():
                if entry["type"] == "pdf":
                    pdf_count += 1
                elif entry["type"] == "summary":
                    summary_count += 1
                total_size += entry.get("file_size_bytes", 0)

                ts = entry.get("cached_at", "")
                if ts:
                    if oldest is None or ts < oldest:
                        oldest = ts
                    if newest is None or ts > newest:
                        newest = ts

            return {
                "total_pdfs": pdf_count,
                "total_summaries": summary_count,
                "total_entries": len(self._data),
                "cache_size_mb": round(total_size / (1024 * 1024), 2),
                "cache_size_bytes": total_size,
                "oldest_entry": oldest,
                "newest_entry": newest,
            }

    def get_expired_entries(self, older_than_days: int = 30) -> List[str]:
        """获取过期的条目 key 列表"""
        threshold = datetime.now() - timedelta(days=older_than_days)
        expired = []
        with self._lock:
            for key, entry in self._data.items():
                ts = entry.get("cached_at", "")
                if ts:
                    try:
                        cached_time = datetime.fromisoformat(ts)
                        if cached_time < threshold:
                            expired.append(key)
                    except ValueError:
                        pass
        return expired


# ============================================================
# 缓存管理器
# ============================================================

class PaperCache:
    """
    论文缓存管理器。

    使用示例:
        cache = PaperCache()

        # PDF 缓存
        pdf_path = cache.get_pdf_path("10.1038/s42256-025-01234-5")
        if pdf_path is None:
            # 下载 PDF...
            pdf_path = cache.cache_pdf(doi, pdf_bytes)

        # 摘要缓存
        summary = cache.get_summary(doi)
        if summary is None:
            summary = llm.generate_summary(paper)
            cache.cache_summary(doi, summary)
    """

    def __init__(self, cache_dir: str = "papers_cache"):
        """
        Args:
            cache_dir: 缓存根目录
        """
        self.cache_dir = Path(cache_dir)
        self.summary_dir = self.cache_dir / "summaries"
        self._metadata = CacheMetadata()
        self._lock = threading.Lock()
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        """确保缓存目录存在"""
        ensure_directory(str(self.cache_dir))
        ensure_directory(str(self.summary_dir))

    # --------------------------------------------------------
    # PDF 缓存
    # --------------------------------------------------------

    def get_pdf_path(self, paper_identifier: str) -> Optional[Path]:
        """
        检查 PDF 是否已缓存。

        Args:
            paper_identifier: DOI 或 paper_id

        Returns:
            PDF 文件的 Path，未缓存则返回 None
        """
        doi_safe = _doi_to_safe_filename(paper_identifier)
        pdf_path = self.cache_dir / f"{doi_safe}.pdf"

        if pdf_path.exists() and pdf_path.stat().st_size > 0:
            logger.debug(f"PDF 缓存命中: {doi_safe}")
            return pdf_path

        return None

    def cache_pdf(
        self,
        paper_identifier: str,
        pdf_content: bytes,
        metadata: Optional[Dict] = None,
    ) -> Path:
        """
        缓存 PDF 文件到磁盘。

        Args:
            paper_identifier: DOI 或 paper_id
            pdf_content: PDF 文件的字节内容
            metadata: 额外元数据（如 title, authors）

        Returns:
            缓存的 PDF 文件 Path
        """
        doi_safe = _doi_to_safe_filename(paper_identifier)
        pdf_path = self.cache_dir / f"{doi_safe}.pdf"

        with self._lock:
            pdf_path.write_bytes(pdf_content)

        file_size = pdf_path.stat().st_size

        self._metadata.set(
            doi_safe=doi_safe,
            entry_type="pdf",
            file_path=str(pdf_path),
            file_size=file_size,
            extra=metadata,
        )

        logger.info(f"PDF 已缓存: {doi_safe} ({file_size / 1024:.1f} KB)")
        return pdf_path

    # --------------------------------------------------------
    # 摘要缓存
    # --------------------------------------------------------

    def get_summary(self, paper_identifier: str) -> Optional[str]:
        """
        检查摘要是否已生成并缓存。

        Args:
            paper_identifier: DOI 或 paper_id

        Returns:
            摘要文本，未缓存则返回 None
        """
        doi_safe = _doi_to_safe_filename(paper_identifier)
        summary_path = self.summary_dir / f"{doi_safe}_summary.md"

        if summary_path.exists() and summary_path.stat().st_size > 0:
            logger.debug(f"摘要缓存命中: {doi_safe}")
            return summary_path.read_text(encoding="utf-8")

        return None

    def cache_summary(
        self,
        paper_identifier: str,
        summary: str,
        metadata: Optional[Dict] = None,
    ) -> Path:
        """
        缓存生成的摘要到磁盘。

        Args:
            paper_identifier: DOI 或 paper_id
            summary: Markdown 格式的摘要文本
            metadata: 额外元数据

        Returns:
            缓存的摘要文件 Path
        """
        doi_safe = _doi_to_safe_filename(paper_identifier)
        summary_path = self.summary_dir / f"{doi_safe}_summary.md"

        with self._lock:
            summary_path.write_text(summary, encoding="utf-8")

        file_size = summary_path.stat().st_size

        self._metadata.set(
            doi_safe=doi_safe,
            entry_type="summary",
            file_path=str(summary_path),
            file_size=file_size,
            extra=metadata,
        )

        logger.info(f"摘要已缓存: {doi_safe} ({file_size} bytes)")
        return summary_path

    # --------------------------------------------------------
    # 批量查询
    # --------------------------------------------------------

    def has_pdf(self, paper_identifier: str) -> bool:
        """检查 PDF 是否已缓存"""
        return self.get_pdf_path(paper_identifier) is not None

    def has_summary(self, paper_identifier: str) -> bool:
        """检查摘要是否已缓存"""
        return self.get_summary(paper_identifier) is not None

    # --------------------------------------------------------
    # 统计与维护
    # --------------------------------------------------------

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        返回缓存统计信息。

        Returns:
            {
                "total_pdfs": int,
                "total_summaries": int,
                "total_entries": int,
                "cache_size_mb": float,
            }
        """
        return self._metadata.get_stats()

    def print_stats(self) -> None:
        """格式化打印缓存统计"""
        stats = self.get_cache_stats()
        print(f"\n{'─' * 40}")
        print(f"  论文缓存统计")
        print(f"{'─' * 40}")
        print(f"  PDF 缓存:    {stats['total_pdfs']} 个")
        print(f"  摘要缓存:    {stats['total_summaries']} 个")
        print(f"  总条目:      {stats['total_entries']}")
        print(f"  磁盘占用:    {stats['cache_size_mb']} MB")
        if stats.get("oldest_entry"):
            print(f"  最早缓存:    {stats['oldest_entry'][:19]}")
        if stats.get("newest_entry"):
            print(f"  最新缓存:    {stats['newest_entry'][:19]}")
        print(f"{'─' * 40}\n")

    def clear_cache(self, older_than_days: int = 30) -> int:
        """
        清理超过指定天数的旧缓存。

        Args:
            older_than_days: 保留最近 N 天的缓存

        Returns:
            清理的条目数
        """
        expired_keys = self._metadata.get_expired_entries(older_than_days)
        removed = 0

        for key in expired_keys:
            entry = self._metadata._data.get(key, {})
            file_path = Path(entry.get("file_path", ""))

            # 删除文件
            try:
                if file_path.exists():
                    file_path.unlink()
            except Exception as e:
                logger.warning(f"删除缓存文件失败 {file_path}: {e}")

            # 删除元数据记录
            doi_safe = entry.get("identifier", "")
            entry_type = entry.get("type", "pdf")
            self._metadata.remove(doi_safe, entry_type)
            removed += 1

        if removed > 0:
            logger.info(f"清理了 {removed} 条过期缓存 (>{older_than_days}天)")

        return removed

    def clear_all(self) -> int:
        """清空所有缓存"""
        removed = 0

        # 删除所有 PDF
        for pdf in self.cache_dir.glob("*.pdf"):
            try:
                pdf.unlink()
                removed += 1
            except Exception as e:
                logger.warning(f"删除 PDF 失败 {pdf}: {e}")

        # 删除所有摘要
        for summary in self.summary_dir.glob("*_summary.md"):
            try:
                summary.unlink()
                removed += 1
            except Exception as e:
                logger.warning(f"删除摘要失败 {summary}: {e}")

        # 清空元数据
        self._metadata._data.clear()
        self._metadata._save()

        logger.info(f"已清空所有缓存 ({removed} 个文件)")
        return removed

    # --------------------------------------------------------
    # 磁盘空间
    # --------------------------------------------------------

    def get_disk_usage_mb(self) -> float:
        """获取缓存目录实际磁盘占用（MB）"""
        total = 0
        if self.cache_dir.exists():
            for f in self.cache_dir.rglob("*"):
                if f.is_file():
                    total += f.stat().st_size
        return round(total / (1024 * 1024), 2)


# ============================================================
# 自测入口
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    print(f"\n{'═' * 50}")
    print("  论文缓存管理器 - 自测")
    print(f"{'═' * 50}")

    cache = PaperCache()

    # 测试1: 缓存 PDF
    print("\n[1] 缓存 PDF...")
    test_doi = "10.1038/s42256-025-01234-5"
    test_pdf = b"%PDF-1.4\nThis is a mock PDF content for testing.\n%%EOF"

    # 首次缓存
    path1 = cache.cache_pdf(test_doi, test_pdf, {"title": "Test Paper"})
    print(f"    ✅ PDF 已缓存: {path1}")
    assert cache.has_pdf(test_doi)

    # 检查缓存命中
    path2 = cache.get_pdf_path(test_doi)
    print(f"    ✅ 缓存命中: {path2}")
    assert path2 == path1

    # 检查未缓存
    path3 = cache.get_pdf_path("10.0000/nonexistent")
    print(f"    ✅ 未缓存返回: {path3}")
    assert path3 is None

    # 测试2: 缓存摘要
    print("\n[2] 缓存摘要...")
    test_summary = "# Paper Summary\n\nThis is a test summary for the paper.\n\n## Key Points\n- Point 1\n- Point 2"

    sp1 = cache.cache_summary(test_doi, test_summary)
    print(f"    ✅ 摘要已缓存: {sp1}")
    assert cache.has_summary(test_doi)

    cached_summary = cache.get_summary(test_doi)
    print(f"    ✅ 摘要命中: {cached_summary[:50]}...")
    assert cached_summary == test_summary

    # 测试3: 多标识符支持
    print("\n[3] 多标识符支持...")
    arxiv_id = "arXiv:2506.12345"
    cache.cache_pdf(arxiv_id, b"arxiv pdf content")
    cache_pdf = cache.get_pdf_path(arxiv_id)
    print(f"    ✅ ArXiv ID 缓存: {cache_pdf}")
    assert cache_pdf is not None

    # 测试4: 统计
    print("\n[4] 缓存统计...")
    cache.print_stats()
    stats = cache.get_cache_stats()
    assert stats["total_pdfs"] >= 2
    assert stats["total_summaries"] >= 1

    # 测试5: 清理
    print("\n[5] 清理测试...")
    # 清理所有缓存（自测用）
    removed = cache.clear_all()
    print(f"    ✅ 已清理: {removed} 个文件")

    print(f"\n✅ 自测完成")
