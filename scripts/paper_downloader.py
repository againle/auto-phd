"""
论文自动下载器（Paper Downloader）

功能：
- 多源下载: arXiv → Semantic Scholar → UnPaywall → DOI 直接解析
- 与 PaperCache 集成，避免重复下载
- PDF 文本提取 (pdfplumber / PyPDF2)
- 下载失败时保存元数据作为回退
- tqdm 进度条 + 详细日志

下载优先级:
1. arXiv (免费，最快)
2. Semantic Scholar (开放获取)
3. UnPaywall API (开放获取)
4. DOI 直接跳转 (可能需权限)
"""

import io
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import requests
from tqdm import tqdm

from scripts.paper_cache import PaperCache
from scripts.utils import ensure_directory, save_json

logger = logging.getLogger(__name__)

# ============================================================
# 常量
# ============================================================

ARXIV_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}.pdf"
ARXIV_ABS_URL = "https://arxiv.org/abs/{arxiv_id}"
SEMANTIC_SCHOLAR_PAPER_URL = "https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
UNPAYWALL_URL = "https://api.unpaywall.org/v2/{doi}?email={email}"

DOWNLOAD_TIMEOUT = 30  # 秒
MAX_PDF_SIZE_MB = 50   # 最大 PDF 大小

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36 "
        "(Academic Research; mailto:student@example.edu)"
    ),
    "Accept": "application/pdf,text/html,application/json,*/*",
}


# ============================================================
# 论文下载器
# ============================================================

class PaperDownloader:
    """
    论文自动下载器。

    使用示例:
        cache = PaperCache()
        downloader = PaperDownloader(cache)
        result = downloader.download_papers(professor_dict, max_papers=3)
    """

    def __init__(self, cache: PaperCache, contact_email: str = ""):
        """
        Args:
            cache: PaperCache 实例
            contact_email: 用于 UnPaywall API 的联系邮箱
        """
        self.cache = cache
        self.contact_email = contact_email or "student@example.edu"
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.session.timeout = DOWNLOAD_TIMEOUT

        # 统计
        self._download_stats = {
            "arxiv": 0,
            "semantic_scholar": 0,
            "unpaywall": 0,
            "doi_direct": 0,
            "cached": 0,
            "failed": 0,
        }

        logger.info("论文下载器初始化完成")

    # --------------------------------------------------------
    # 主入口：批量下载教授论文
    # --------------------------------------------------------

    def download_papers(
        self,
        professor: Dict[str, Any],
        max_papers: int = 3,
        prof_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        为教授下载论文（取引用数最高的前N篇）。

        Args:
            professor: 教授信息字典
            max_papers: 最多下载几篇
            prof_dir: 教授文件夹路径（用于保存到 papers/ 子目录）

        Returns:
            {
                "downloaded": [{"title": ..., "path": ..., "source": ...}],
                "cached": [{"title": ..., "path": ...}],
                "failed": [{"title": ..., "reason": ...}],
                "total": int,
            }
        """
        papers = professor.get("recent_papers", [])
        name = professor.get("name", "Unknown")

        # 按引用数排序，取前N篇
        sorted_papers = sorted(
            papers,
            key=lambda p: p.get("citations", 0),
            reverse=True,
        )[:max_papers]

        if not sorted_papers:
            logger.warning(f"教授 {name} 无可用论文")
            return {"downloaded": [], "cached": [], "failed": [], "total": 0}

        logger.info(f"开始为 {name} 下载论文 (最多 {max_papers} 篇)")

        result = {
            "downloaded": [],
            "cached": [],
            "failed": [],
            "total": len(sorted_papers),
        }

        # 确定保存目录
        if prof_dir:
            papers_dir = Path(prof_dir) / "papers"
        else:
            papers_dir = Path("papers_cache")
        ensure_directory(str(papers_dir))

        for paper in tqdm(sorted_papers, desc=f"下载 {name[:20]}", unit="paper", ncols=80):
            title = paper.get("title", "Unknown")
            paper_id = paper.get("paper_id", "")
            doi = paper.get("doi", "")
            ext_ids = paper.get("external_ids", {}) or {}

            # 尝试从 external_ids 获取更多标识符
            if not doi:
                doi = ext_ids.get("DOI", "")
            arxiv_id = ext_ids.get("ArXiv", "")

            try:
                pdf_path = self._download_single_paper(
                    title=title,
                    paper_id=paper_id,
                    doi=doi,
                    arxiv_id=arxiv_id,
                    output_dir=papers_dir,
                )

                if pdf_path:
                    if pdf_path.name.endswith(".json"):
                        # 只有元数据（PDF 下载失败的回退）
                        result["failed"].append({
                            "title": title,
                            "reason": "PDF 不可获取，已保存元数据",
                            "metadata_path": str(pdf_path),
                        })
                    else:
                        # 判断是缓存命中还是新下载
                        source = "cached" if self._is_from_cache(pdf_path) else "downloaded"
                        result[source].append({
                            "title": title,
                            "path": str(pdf_path),
                            "source": source,
                        })
                else:
                    result["failed"].append({
                        "title": title,
                        "reason": "所有下载源均不可用",
                    })

            except Exception as e:
                logger.error(f"下载论文失败 '{title[:50]}...': {e}")
                result["failed"].append({
                    "title": title,
                    "reason": str(e)[:100],
                })

        # 汇总
        total_ok = len(result["downloaded"]) + len(result["cached"])
        logger.info(
            f"{name} 论文下载完成: "
            f"下载 {len(result['downloaded'])}, "
            f"缓存 {len(result['cached'])}, "
            f"失败 {len(result['failed'])}"
        )

        return result

    # --------------------------------------------------------
    # 单篇下载（多源尝试）
    # --------------------------------------------------------

    def _download_single_paper(
        self,
        title: str,
        paper_id: str = "",
        doi: str = "",
        arxiv_id: str = "",
        output_dir: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        尝试多个源下载单篇论文。

        优先级: arXiv > Semantic Scholar > UnPaywall > DOI

        Returns:
            PDF 路径、元数据 JSON 路径、或 None
        """
        if output_dir is None:
            output_dir = Path("papers_cache")

        # 生成安全的文件名前缀
        safe_title = re.sub(r'[<>:"/\\|?*]', '', title)[:60]
        ident = doi or arxiv_id or paper_id or safe_title

        # 1. 检查缓存
        cached_pdf = self.cache.get_pdf_path(ident)
        if cached_pdf:
            self._download_stats["cached"] += 1
            logger.debug(f"缓存命中: {ident}")
            return cached_pdf

        pdf_content: Optional[bytes] = None
        download_source = ""

        # 2. 尝试 arXiv
        if arxiv_id:
            pdf_content = self._try_download_arxiv(arxiv_id)
            if pdf_content:
                download_source = "arxiv"
                self._download_stats["arxiv"] += 1

        # 3. 尝试 Semantic Scholar
        if not pdf_content and paper_id:
            pdf_content = self._try_download_semantic_scholar(paper_id)
            if pdf_content:
                download_source = "semantic_scholar"
                self._download_stats["semantic_scholar"] += 1

        # 4. 尝试 UnPaywall
        if not pdf_content and doi:
            pdf_content = self._try_download_unpaywall(doi)
            if pdf_content:
                download_source = "unpaywall"
                self._download_stats["unpaywall"] += 1

        # 5. 尝试 DOI 直接解析
        if not pdf_content and doi:
            pdf_content = self._try_download_doi_direct(doi)
            if pdf_content:
                download_source = "doi_direct"
                self._download_stats["doi_direct"] += 1

        # 成功：缓存 PDF
        if pdf_content:
            # 验证 PDF 有效性
            if not pdf_content.startswith(b"%PDF"):
                logger.warning(f"下载的内容不是有效 PDF: {ident}")
                pdf_content = None

        if pdf_content:
            cached_path = self.cache.cache_pdf(
                ident,
                pdf_content,
                metadata={
                    "title": title,
                    "doi": doi,
                    "arxiv_id": arxiv_id,
                    "source": download_source,
                },
            )
            logger.info(f"✅ 下载成功 [{download_source}]: {title[:60]}")
            return cached_path

        # 失败：保存元数据作为回退
        self._download_stats["failed"] += 1
        logger.warning(f"❌ 所有下载源均失败: {title[:60]}")

        meta_path = output_dir / f"{safe_title[:40]}_metadata.json"
        save_json({
            "title": title,
            "doi": doi,
            "arxiv_id": arxiv_id,
            "paper_id": paper_id,
            "download_failed": True,
            "failed_at": datetime.now().isoformat(),
        }, str(meta_path))

        return meta_path

    # --------------------------------------------------------
    # 各下载源实现
    # --------------------------------------------------------

    def _try_download_arxiv(self, arxiv_id: str) -> Optional[bytes]:
        """尝试从 arXiv 下载 PDF"""
        # 清理 arXiv ID（去掉 "arxiv:" 前缀和版本号）
        clean_id = arxiv_id.replace("arxiv:", "").replace("arXiv:", "").strip()
        clean_id = re.sub(r"v\d+$", "", clean_id)

        url = ARXIV_PDF_URL.format(arxiv_id=clean_id)
        logger.debug(f"尝试 arXiv: {url}")

        try:
            resp = self.session.get(url, timeout=DOWNLOAD_TIMEOUT)
            if resp.status_code == 200 and len(resp.content) > 1000:
                return resp.content
            else:
                logger.debug(f"arXiv 返回 {resp.status_code}, 大小 {len(resp.content)}")
                return None
        except Exception as e:
            logger.debug(f"arXiv 下载失败: {e}")
            return None

    def _try_download_semantic_scholar(self, paper_id: str) -> Optional[bytes]:
        """尝试通过 Semantic Scholar API 获取开放获取 PDF"""
        url = SEMANTIC_SCHOLAR_PAPER_URL.format(paper_id=paper_id)
        params = {
            "fields": "openAccessPdf,title",
        }

        logger.debug(f"尝试 Semantic Scholar: {paper_id}")

        try:
            resp = self.session.get(url, params=params, timeout=DOWNLOAD_TIMEOUT)
            if resp.status_code != 200:
                return None

            data = resp.json()
            oa_info = data.get("openAccessPdf") or {}
            pdf_url = oa_info.get("url", "")

            if not pdf_url:
                logger.debug("Semantic Scholar: 无开放获取 PDF")
                return None

            # 下载 PDF
            pdf_resp = self.session.get(pdf_url, timeout=DOWNLOAD_TIMEOUT)
            if pdf_resp.status_code == 200 and len(pdf_resp.content) > 1000:
                return pdf_resp.content

            return None
        except Exception as e:
            logger.debug(f"Semantic Scholar 下载失败: {e}")
            return None

    def _try_download_unpaywall(self, doi: str) -> Optional[bytes]:
        """尝试通过 UnPaywall API 获取开放获取 PDF"""
        clean_doi = doi.strip().lower()
        url = UNPAYWALL_URL.format(doi=clean_doi, email=self.contact_email)

        logger.debug(f"尝试 UnPaywall: {clean_doi}")

        try:
            resp = self.session.get(url, timeout=DOWNLOAD_TIMEOUT)
            if resp.status_code != 200:
                return None

            data = resp.json()
            best_oa = data.get("best_oa_location") or {}
            pdf_url = best_oa.get("url_for_pdf", "")

            if not pdf_url:
                # 尝试其他 OA 位置
                for loc in data.get("oa_locations", []):
                    pdf_url = loc.get("url_for_pdf", "")
                    if pdf_url:
                        break

            if not pdf_url:
                logger.debug("UnPaywall: 无开放获取 PDF")
                return None

            pdf_resp = self.session.get(pdf_url, timeout=DOWNLOAD_TIMEOUT)
            if pdf_resp.status_code == 200 and len(pdf_resp.content) > 1000:
                return pdf_resp.content

            return None
        except Exception as e:
            logger.debug(f"UnPaywall 下载失败: {e}")
            return None

    def _try_download_doi_direct(self, doi: str) -> Optional[bytes]:
        """尝试通过 DOI 直接跳转页面寻找 PDF"""
        clean_doi = doi.strip()
        doi_url = f"https://doi.org/{clean_doi}"

        logger.debug(f"尝试 DOI 直接访问: {clean_doi}")

        try:
            # 获取重定向后的页面
            resp = self.session.get(
                doi_url,
                headers={**HEADERS, "Accept": "application/pdf,text/html"},
                allow_redirects=True,
                timeout=DOWNLOAD_TIMEOUT,
            )

            # 如果直接返回 PDF
            if "application/pdf" in resp.headers.get("Content-Type", ""):
                if len(resp.content) > 1000:
                    return resp.content

            # 尝试在 HTML 中找 PDF 链接
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" in content_type:
                text = resp.text.lower()
                # 常见模式
                pdf_patterns = [
                    r'href="([^"]+\.pdf)"',
                    r'href=\'([^\']+\.pdf)\'',
                    r'<meta[^>]+citation_pdf_url["\s]+content="([^"]+)"',
                ]
                for pattern in pdf_patterns:
                    matches = re.findall(pattern, resp.text, re.IGNORECASE)
                    for match in matches:
                        if not match.startswith("http"):
                            # 相对 URL
                            from urllib.parse import urljoin
                            match = urljoin(resp.url, match)
                        try:
                            pdf_resp = self.session.get(match, timeout=DOWNLOAD_TIMEOUT)
                            if pdf_resp.status_code == 200 and len(pdf_resp.content) > 1000:
                                if b"PDF" in pdf_resp.content[:100] or b"%PDF" in pdf_resp.content[:100]:
                                    return pdf_resp.content
                        except Exception:
                            continue

            return None
        except Exception as e:
            logger.debug(f"DOI 直接下载失败: {e}")
            return None

    # --------------------------------------------------------
    # 便捷方法
    # --------------------------------------------------------

    def download_by_doi(self, doi: str, output_dir: Optional[Path] = None) -> Optional[Path]:
        """通过 DOI 下载论文（公开接口）"""
        return self._download_single_paper(
            title=f"DOI:{doi}",
            doi=doi,
            output_dir=output_dir or Path("papers_cache"),
        )

    def download_by_arxiv(self, arxiv_id: str, output_dir: Optional[Path] = None) -> Optional[Path]:
        """通过 arXiv ID 下载论文（公开接口）"""
        return self._download_single_paper(
            title=f"arXiv:{arxiv_id}",
            arxiv_id=arxiv_id,
            output_dir=output_dir or Path("papers_cache"),
        )

    def download_by_semantic_scholar(self, paper_id: str, output_dir: Optional[Path] = None) -> Optional[Path]:
        """通过 Semantic Scholar 下载论文（公开接口）"""
        return self._download_single_paper(
            title=f"S2:{paper_id}",
            paper_id=paper_id,
            output_dir=output_dir or Path("papers_cache"),
        )

    # --------------------------------------------------------
    # PDF 文本提取
    # --------------------------------------------------------

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        从 PDF 提取文本。

        优先使用 pdfplumber（保留布局），
        回退到 PyPDF2（兼容性好）。

        Args:
            pdf_path: PDF 文件路径

        Returns:
            提取的文本（如果全部失败返回空字符串）
        """
        if not pdf_path.exists():
            logger.error(f"PDF 文件不存在: {pdf_path}")
            return ""

        # 1. 尝试 pdfplumber
        try:
            import pdfplumber
            text_parts = []
            with pdfplumber.open(str(pdf_path)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)

            text = "\n\n".join(text_parts)
            if len(text.strip()) > 100:
                logger.debug(f"pdfplumber 提取成功: {len(text)} 字符")
                return text
        except Exception as e:
            logger.debug(f"pdfplumber 提取失败: {e}")

        # 2. 回退到 PyPDF2
        try:
            from PyPDF2 import PdfReader
            text_parts = []
            reader = PdfReader(str(pdf_path))
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

            text = "\n\n".join(text_parts)
            if len(text.strip()) > 50:
                logger.debug(f"PyPDF2 提取成功: {len(text)} 字符")
                return text
        except Exception as e:
            logger.debug(f"PyPDF2 提取失败: {e}")

        # 3. 最后尝试：读取 raw bytes 中的文本
        try:
            raw = pdf_path.read_bytes()
            # 尝试提取 ASCII 文本段
            text_chars = []
            for byte in raw:
                if 32 <= byte < 127 or byte in (10, 13):
                    text_chars.append(chr(byte))
            text = "".join(text_chars)
            if len(text.strip()) > 100:
                logger.debug(f"Raw 文本提取: {len(text)} 字符")
                return text
        except Exception as e:
            logger.debug(f"Raw 提取失败: {e}")

        logger.warning(f"无法从 PDF 提取文本: {pdf_path.name}")
        return ""

    # --------------------------------------------------------
    # 辅助方法
    # --------------------------------------------------------

    def _is_from_cache(self, pdf_path: Path) -> bool:
        """判断 PDF 是否来自缓存（通过检查缓存目录）"""
        cache_dir = str(self.cache.cache_dir)
        return cache_dir in str(pdf_path)

    def get_download_stats(self) -> Dict[str, int]:
        """返回下载统计"""
        return dict(self._download_stats)

    def reset_stats(self) -> None:
        """重置下载统计"""
        for key in self._download_stats:
            self._download_stats[key] = 0


# ============================================================
# 自测入口
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    print(f"\n{'═' * 60}")
    print("  论文下载器 - 自测")
    print(f"{'═' * 60}")

    cache = PaperCache()
    downloader = PaperDownloader(cache)

    # 测试1: arXiv 下载
    print("\n[1] 测试 arXiv 下载...")
    # 使用一篇知名的公开论文
    arxiv_id = "1706.03762"  # "Attention Is All You Need"
    result = downloader.download_by_arxiv(arxiv_id)
    if result and result.suffix == ".pdf":
        print(f"    ✅ arXiv 下载成功: {result.name}")
        # 测试文本提取
        text = downloader.extract_text_from_pdf(result)
        print(f"    ✅ 文本提取: {len(text)} 字符")
        print(f"    开头: {text[:100]}...")
    else:
        print(f"    ⚠️ arXiv 下载失败（网络问题或限流）: {result}")

    # 测试2: 模拟教授论文下载
    print("\n[2] 模拟教授论文下载...")
    mock_professor = {
        "name": "Test Professor",
        "recent_papers": [
            {
                "title": "Attention Is All You Need",
                "year": 2017,
                "citations": 100000,
                "paper_id": "",
                "external_ids": {"ArXiv": "1706.03762"},
            },
            {
                "title": "Non-existent Paper",
                "year": 2025,
                "citations": 1,
                "paper_id": "nonexistent12345",
                "doi": "10.9999/nonexistent",
            },
        ],
    }

    result = downloader.download_papers(mock_professor, max_papers=2)
    print(f"    下载成功: {len(result['downloaded'])}")
    print(f"    缓存命中: {len(result['cached'])}")
    print(f"    失败: {len(result['failed'])}")
    for fail in result["failed"]:
        print(f"      - {fail['title'][:50]}: {fail['reason']}")

    # 测试3: 统计
    print("\n[3] 下载统计...")
    stats = downloader.get_download_stats()
    for source, count in stats.items():
        if count > 0:
            print(f"    {source}: {count}")

    cache.print_stats()

    print(f"\n✅ 自测完成")
