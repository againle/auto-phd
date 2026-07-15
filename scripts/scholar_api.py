"""
Semantic Scholar API 封装

功能：
- 按关键词搜索教授（作者搜索 + 论文检索）
- 获取论文详细信息
- 引用追溯（获取引用某论文的其他论文）
- 内置缓存机制（1小时TTL）+ 请求限流
- 基于 tenacity 的智能重试

API文档: https://api.semanticscholar.org/api-docs
"""

import hashlib
import json
import logging
import time
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Any

import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from scripts.utils import ensure_directory, load_config, load_json, save_json

logger = logging.getLogger(__name__)

# ============================================================
# 常量
# ============================================================

BASE_URL = "https://api.semanticscholar.org/graph/v1"
CACHE_FILE = "papers_cache/api_cache.json"
CACHE_TTL_HOURS = 1

# 请求间隔（秒），避免限流（免费版限制 ~1 req/s）
MIN_DELAY = 1.0
MAX_DELAY = 1.5

# 通用请求头
HEADERS = {
    "User-Agent": "AutoPhD-Academic-Application/1.0 (mailto:student@example.edu)",
    "Accept": "application/json",
}

# 作者搜索所需的字段
AUTHOR_SEARCH_FIELDS = [
    "name",
    "affiliations",
    "paperCount",
    "citationCount",
    "hIndex",
    "url",
    "externalIds",
]

# 论文详情所需的字段
PAPER_FIELDS = [
    "title",
    "abstract",
    "authors",
    "year",
    "citationCount",
    "influentialCitationCount",
    "publicationVenue",
    "externalIds",
    "url",
    "openAccessPdf",
    "fieldsOfStudy",
    "tldr",
    "citationStyles",
]

# 引用论文所需的字段
CITATION_FIELDS = [
    "title",
    "year",
    "citationCount",
    "authors",
    "abstract",
    "url",
    "externalIds",
]


# ============================================================
# 缓存管理
# ============================================================

class QueryCache:
    """
    简单的查询缓存，基于 JSON 文件存储。

    缓存键 = 查询参数的 SHA256 哈希
    缓存有效期 = CACHE_TTL_HOURS
    """

    def __init__(self, cache_path: str = CACHE_FILE):
        self._cache_path = Path(cache_path)
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        """从文件加载缓存"""
        if self._cache_path.exists():
            try:
                self._cache = load_json(str(self._cache_path))
                # 清理过期条目
                expired = []
                now = datetime.now().isoformat()
                for key, entry in self._cache.items():
                    if entry.get("expires_at", "") < now:
                        expired.append(key)
                for key in expired:
                    del self._cache[key]
                if expired:
                    logger.debug(f"清理了 {len(expired)} 条过期缓存")
            except Exception as e:
                logger.warning(f"加载缓存失败: {e}，将使用空缓存")
                self._cache = {}

    def _save(self) -> None:
        """保存缓存到文件"""
        try:
            ensure_directory(str(self._cache_path.parent))
            save_json(self._cache, str(self._cache_path))
        except Exception as e:
            logger.warning(f"保存缓存失败: {e}")

    def _make_key(self, *args: Any) -> str:
        """生成缓存键"""
        raw = json.dumps(args, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get(self, *args: Any) -> Optional[Any]:
        """获取缓存值，过期返回 None"""
        key = self._make_key(*args)
        entry = self._cache.get(key)
        if entry and entry.get("expires_at", "") > datetime.now().isoformat():
            logger.debug(f"缓存命中: {key}")
            return entry["data"]
        return None

    def set(self, data: Any, *args: Any, ttl_hours: int = CACHE_TTL_HOURS) -> None:
        """设置缓存值"""
        key = self._make_key(*args)
        self._cache[key] = {
            "data": data,
            "expires_at": (datetime.now() + timedelta(hours=ttl_hours)).isoformat(),
        }
        self._save()
        logger.debug(f"缓存写入: {key}")


# 全局缓存实例
_cache = QueryCache()


# ============================================================
# 请求工具
# ============================================================

def _rate_limit() -> None:
    """请求间隔延迟，避免触发 API 限流"""
    delay = random.uniform(MIN_DELAY, MAX_DELAY)
    time.sleep(delay)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=3, max=60),
    retry=retry_if_exception_type((
        requests.exceptions.Timeout,
        requests.exceptions.ConnectionError,
        requests.exceptions.HTTPError,
    )),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _get(url: str, params: Optional[Dict] = None, timeout: int = 15) -> Dict:
    """
    发送 GET 请求（带重试和限流）。

    Args:
        url: 请求 URL
        params: 查询参数
        timeout: 超时秒数

    Returns:
        JSON 响应字典

    Raises:
        requests.RequestException: 请求失败时抛出
    """
    _rate_limit()

    response = requests.get(
        url,
        params=params,
        headers=HEADERS,
        timeout=timeout,
    )

    # 429 Too Many Requests → 等待并重试
    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After", "10")
        wait_seconds = int(retry_after) if retry_after.isdigit() else 10
        logger.warning(f"API 限流 (429)，等待 {wait_seconds}s 后重试...")
        time.sleep(wait_seconds)
        response.raise_for_status()  # 触发 HTTPError → tenacity 重试

    response.raise_for_status()
    return response.json()


# ============================================================
# 教授搜索（论文→作者策略）
# ============================================================

def search_professors(
    query: str,
    limit: int = 50,
    year: Optional[int] = 3,
) -> List[Dict[str, Any]]:
    """
    根据关键词搜索教授。

    策略：先搜论文 → 提取作者 → 聚合作者信息。
    这比直接搜作者名更精准，因为按研究方向匹配。

    Args:
        query: 搜索关键词（如 "protein structure deep learning"）
        limit: 最大返回结果数
        year: 只看近N年发表的作者，None 表示不限

    Returns:
        教授列表，每个包含：
        - name, institution, research_topics,
        - publication_count, h_index, recent_papers,
        - scholar_id, url
    """
    cached = _cache.get("search_professors", query, limit, year)
    if cached is not None:
        logger.info(f"使用缓存结果: query='{query}'")
        return cached

    logger.info(f"搜索教授: query='{query}', limit={limit}, year_filter={year}")

    # 第一步：按关键词搜索论文
    papers = _search_papers(query, limit=min(limit * 3, 100), year=year)
    if not papers:
        logger.warning(f"未找到匹配论文: '{query}'")
        return []

    # 第二步：从论文中提取唯一作者
    author_map: Dict[str, Dict] = {}  # {author_id: {name, paper_count, papers}}
    for paper in papers:
        for author in paper.get("authors", []):
            aid = author.get("authorId", "")
            if not aid:
                continue
            if aid not in author_map:
                author_map[aid] = {
                    "name": author.get("name", "Unknown"),
                    "paper_count": 0,
                    "papers": [],
                }
            author_map[aid]["paper_count"] += 1
            author_map[aid]["papers"].append({
                "title": paper.get("title", ""),
                "year": paper.get("year"),
                "citations": paper.get("citationCount", 0),
                "venue": paper.get("publicationVenue") or "",
                "paper_id": paper.get("paperId", ""),
            })

    # 第三步：获取作者详细信息（仅 h-index 和 affiliation，不重复获取论文）
    professors = []
    year_threshold = datetime.now().year - year if year else 0

    for aid, info in list(author_map.items())[:limit]:
        try:
            # 只需一次 API 调用获取作者元信息
            author_detail = _get_author_detail(aid)

            # 研究方向从已有论文标题提取（无需额外 API 调用）
            research_topics = _extract_topics_from_papers(info["papers"])

            # 按年份过滤论文
            recent_papers = info["papers"]
            if year_threshold:
                recent_papers = [p for p in info["papers"] if (p.get("year") or 0) >= year_threshold]

            affiliation = ""
            if author_detail.get("affiliations"):
                affiliation = author_detail["affiliations"][0]

            professors.append({
                "name": info["name"],
                "institution": affiliation or "Unknown",
                "email": "",
                "research_topics": research_topics,
                "publication_count": author_detail.get("paperCount", info["paper_count"]),
                "h_index": author_detail.get("hIndex", 0),
                "citation_count": author_detail.get("citationCount", 0),
                "recent_papers": recent_papers[:10],
                "scholar_id": aid,
                "url": author_detail.get("url", f"https://api.semanticscholar.org/author/{aid}"),
            })

        except Exception as e:
            logger.warning(f"获取作者 {info['name']} 详情失败: {e}")
            # 回退：使用论文中已有的信息
            recent_papers = info["papers"]
            if year_threshold:
                recent_papers = [p for p in info["papers"] if (p.get("year") or 0) >= year_threshold]

            professors.append({
                "name": info["name"],
                "institution": "Unknown",
                "email": "",
                "research_topics": _extract_topics_from_papers(info["papers"]),
                "publication_count": info["paper_count"],
                "h_index": 0,
                "citation_count": 0,
                "recent_papers": recent_papers[:10],
                "scholar_id": aid,
                "url": f"https://api.semanticscholar.org/author/{aid}",
            })

    _cache.set(professors, "search_professors", query, limit, year)

    logger.info(
        f"搜索完成: query='{query}' → {len(professors)} 位教授 "
        f"(从 {len(papers)} 篇论文中提取 {len(author_map)} 位作者)"
    )
    return professors


def _search_papers(query: str, limit: int = 100, year: Optional[int] = None) -> List[Dict]:
    """搜索论文（按关键词 + 年份过滤）"""
    url = f"{BASE_URL}/paper/search"
    params = {
        "query": query,
        "limit": min(limit, 100),
        "fields": "title,year,citationCount,authors,publicationVenue,paperId",
    }
    if year:
        current_year = datetime.now().year
        params["year"] = f"{current_year - year}-{current_year}"

    try:
        data = _get(url, params=params)
        papers = data.get("data", [])
        logger.debug(f"论文搜索 '{query}': 找到 {len(papers)} 篇")
        return papers
    except Exception as e:
        logger.error(f"论文搜索失败 '{query}': {e}")
        return []


def _get_author_detail(author_id: str) -> Dict:
    """获取作者详细信息"""
    url = f"{BASE_URL}/author/{author_id}"
    params = {"fields": "name,affiliations,paperCount,citationCount,hIndex,url"}
    try:
        return _get(url, params=params)
    except Exception:
        return {}


def _get_author_papers(author_id: str, year_threshold: int = 0, max_papers: int = 20) -> List[Dict]:
    """获取作者的论文列表（按引用数排序，取最近的）"""
    url = f"{BASE_URL}/author/{author_id}/papers"
    params = {
        "limit": min(max_papers, 100),
        "fields": "title,year,citationCount,abstract,externalIds,url,publicationVenue",
    }

    try:
        data = _get(url, params=params)
        all_papers = data.get("data", [])

        # 按年份过滤
        if year_threshold > 0:
            all_papers = [p for p in all_papers if (p.get("year") or 0) >= year_threshold]

        # 只返回关键字段
        return [
            {
                "title": p.get("title", ""),
                "year": p.get("year"),
                "citations": p.get("citationCount", 0),
                "venue": p.get("publicationVenue") or "",
                "abstract": (p.get("abstract") or "")[:300],  # 截断摘要
                "paper_id": p.get("paperId", ""),
            }
            for p in all_papers
        ]
    except Exception as e:
        logger.error(f"获取作者 {author_id} 论文失败: {e}")
        return []


def _extract_topics_from_papers(papers: List[Dict], max_topics: int = 5) -> List[str]:
    """从论文标题和摘要中提取研究方向关键词（简化版：基于标题分词）"""
    import re

    # 简单的关键词提取：统计标题中的高频词
    word_freq: Dict[str, int] = {}
    stop_words = {
        "a", "an", "the", "and", "or", "of", "in", "on", "to", "for",
        "with", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "can", "shall",
        "this", "that", "these", "those", "it", "its",
        "from", "by", "at", "as", "into", "through", "during",
        "using", "based", "via", "approach", "method", "model",
        "novel", "new", "towards", "learning", "deep", "neural",
        "network", "networks", "data", "analysis", "study",
    }

    for paper in papers:
        title = paper.get("title", "")
        # 提取单词（2字母以上）
        words = re.findall(r"[a-zA-Z]{3,}", title.lower())
        for w in words:
            if w not in stop_words:
                word_freq[w] = word_freq.get(w, 0) + 1

    # 取高频词作为研究方向
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_words[:max_topics]]


# ============================================================
# 论文检索
# ============================================================

def get_paper_details(paper_id: str) -> Dict[str, Any]:
    """
    获取论文详细信息。

    Args:
        paper_id: Semantic Scholar Paper ID (或 DOI, ArXiv ID)

    Returns:
        论文详情字典，包含 title, abstract, authors, year, citationCount 等
    """
    # 检查缓存
    cached = _cache.get("paper_details", paper_id)
    if cached is not None:
        return cached

    logger.info(f"获取论文详情: {paper_id}")

    # 支持 DOI 和 ArXiv ID
    if paper_id.startswith("10."):
        url = f"{BASE_URL}/paper/DOI:{paper_id}"
    elif paper_id.startswith("arXiv:"):
        url = f"{BASE_URL}/paper/ArXiv:{paper_id.split(':')[-1]}"
    else:
        url = f"{BASE_URL}/paper/{paper_id}"

    params = {"fields": ",".join(PAPER_FIELDS)}

    try:
        data = _get(url, params=params)

        # 提取作者信息
        authors = []
        for author in data.get("authors", []):
            authors.append({
                "name": author.get("name", ""),
                "author_id": author.get("authorId", ""),
            })

        # 提取 TL;DR（Semantic Scholar 的自动摘要）
        tldr = data.get("tldr", {}) or {}

        result = {
            "paper_id": data.get("paperId", paper_id),
            "title": data.get("title", ""),
            "abstract": data.get("abstract", ""),
            "tldr": tldr.get("text", ""),
            "authors": authors,
            "year": data.get("year"),
            "citation_count": data.get("citationCount", 0),
            "influential_citations": data.get("influentialCitationCount", 0),
            "venue": data.get("publicationVenue") or {},
            "fields_of_study": data.get("fieldsOfStudy", []),
            "external_ids": data.get("externalIds", {}),
            "url": data.get("url", ""),
            "open_access_pdf": (data.get("openAccessPdf") or {}).get("url", ""),
            "bibtex": (data.get("citationStyles") or {}).get("bibtex", ""),
        }

        _cache.set(result, "paper_details", paper_id)
        logger.info(f"论文详情获取成功: '{result['title'][:60]}...'")
        return result

    except Exception as e:
        logger.error(f"获取论文详情失败 '{paper_id}': {e}")
        return {"paper_id": paper_id, "error": str(e)}


# ============================================================
# 引用追溯
# ============================================================

def get_citing_papers(paper_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    获取引用某篇论文的其他论文（用于扩展搜索、发现相关教授）。

    Args:
        paper_id: Semantic Scholar Paper ID
        limit: 返回数量

    Returns:
        引用论文列表，每篇含 title, year, citationCount, authors
    """
    cached = _cache.get("citing_papers", paper_id, limit)
    if cached is not None:
        return cached

    logger.info(f"获取引用论文: {paper_id}, limit={limit}")

    url = f"{BASE_URL}/paper/{paper_id}/citations"
    params = {
        "limit": min(limit, 100),
        "fields": ",".join(CITATION_FIELDS),
    }

    try:
        data = _get(url, params=params)
        citations = data.get("data", [])

        results = []
        for item in citations:
            paper = item.get("citingPaper", {}) or item
            authors = []
            for author in paper.get("authors", []):
                authors.append({
                    "name": author.get("name", ""),
                    "author_id": author.get("authorId", ""),
                })

            results.append({
                "paper_id": paper.get("paperId", ""),
                "title": paper.get("title", ""),
                "year": paper.get("year"),
                "citation_count": paper.get("citationCount", 0),
                "authors": authors,
                "abstract": (paper.get("abstract") or "")[:300],
                "url": paper.get("url", ""),
            })

        _cache.set(results, "citing_papers", paper_id, limit)
        logger.info(f"引用追溯完成: {len(results)} 篇引用论文")
        return results

    except Exception as e:
        logger.error(f"获取引用论文失败 '{paper_id}': {e}")
        return []


# ============================================================
# 批量搜索（组合多个查询词）
# ============================================================

def batch_search_professors(
    queries: List[str],
    limit_per_query: int = 20,
    year: Optional[int] = 3,
    deduplicate: bool = True,
) -> List[Dict[str, Any]]:
    """
    批量搜索教授，使用多个查询词组合覆盖不同角度。

    Args:
        queries: 搜索关键词列表
        limit_per_query: 每个查询的最大结果数
        year: 只看近N年论文
        deduplicate: 是否按 scholar_id 去重

    Returns:
        合并后的教授列表
    """
    all_professors: List[Dict] = []
    seen_ids: set = set()

    for query in queries:
        logger.info(f"批量搜索 [{len(queries)}]: '{query}'")
        try:
            results = search_professors(query, limit=limit_per_query, year=year)
            for prof in results:
                sid = prof.get("scholar_id", "")
                if deduplicate and sid in seen_ids:
                    continue
                seen_ids.add(sid)
                all_professors.append(prof)
        except Exception as e:
            logger.error(f"批量搜索 '{query}' 失败: {e}")
            continue

    logger.info(f"批量搜索完成: {len(queries)} 个查询 → {len(all_professors)} 位教授（去重后）")
    return all_professors


# ============================================================
# 自测入口
# ============================================================

if __name__ == "__main__":
    # 设置简单日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 60)
    print("  Semantic Scholar API 自测")
    print("=" * 60)

    # 测试1: 搜索教授
    print("\n[1] 搜索教授: 'protein structure deep learning'")
    professors = search_professors("protein structure deep learning", limit=5, year=3)
    for i, prof in enumerate(professors, 1):
        print(f"\n  {i}. {prof['name']}")
        print(f"     机构: {prof['institution']}")
        print(f"     发表: {prof['publication_count']} 篇 (h-index: {prof['h_index']})")
        print(f"     方向: {', '.join(prof['research_topics'][:5])}")
        recent = prof.get("recent_papers", [])[:3]
        for rp in recent:
            print(f"       📄 {rp.get('title', '?')[:60]} ({rp.get('year')})")

    # 测试2: 论文详情（使用搜索到的第一篇论文）
    if professors and professors[0].get("recent_papers"):
        paper_id = professors[0]["recent_papers"][0].get("paper_id", "")
        if paper_id:
            print(f"\n[2] 获取论文详情: {paper_id}")
            paper = get_paper_details(paper_id)
            print(f"  标题: {paper.get('title', '?')[:80]}")
            print(f"  年份: {paper.get('year')}, 引用: {paper.get('citation_count')}")
            print(f"  作者: {', '.join(a['name'] for a in paper.get('authors', [])[:5])}")

    # 测试3: 引用追溯
    if professors and professors[0].get("recent_papers"):
        paper_id = professors[0]["recent_papers"][0].get("paper_id", "")
        if paper_id:
            print(f"\n[3] 引用追溯: {paper_id}")
            citing = get_citing_papers(paper_id, limit=5)
            for j, cp in enumerate(citing, 1):
                print(f"  {j}. {cp.get('title', '?')[:60]} ({cp.get('year')})")

    print("\n✅ 自测完成")
