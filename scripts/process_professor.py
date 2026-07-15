"""
教授处理器（Professor Processor）

提供统一接口，完成单个教授的完整处理流程：
1. 读取 info.json
2. 下载论文 (PaperDownloader)
3. 生成摘要 (PaperReader)
4. 更新状态
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from tqdm import tqdm

from scripts.paper_downloader import PaperDownloader
from scripts.paper_reader import PaperReader
from scripts.utils import load_json, save_json, ensure_directory

logger = logging.getLogger(__name__)

# ============================================================
# 状态定义
# ============================================================

STATUS_TRANSITIONS = {
    "pending_score": "scored",
    "scored": "papers_downloading",
    "papers_downloading": "papers_reading",
    "papers_reading": "paper_reading_completed",
    "paper_reading_completed": "ready_for_email",
    "ready_for_email": "email_drafted",
    "email_drafted": "sent",
}


# ============================================================
# 教授处理器
# ============================================================

class ProfessorProcessor:
    """
    教授处理器 — 统一处理单个教授的完整流水线。

    使用示例:
        processor = ProfessorProcessor(downloader, reader)
        result = processor.process(Path("professors/MIT_Alice/"))
        print(result["status"])
    """

    def __init__(
        self,
        downloader: PaperDownloader,
        reader: PaperReader,
    ):
        """
        Args:
            downloader: PaperDownloader 实例
            reader: PaperReader 实例
        """
        self.downloader = downloader
        self.reader = reader

        # 配置
        self.max_papers_per_professor = 3

        # 统计
        self._stats = {
            "total_processed": 0,
            "total_success": 0,
            "total_failed": 0,
            "total_papers_downloaded": 0,
            "total_papers_read": 0,
            "total_time_seconds": 0.0,
        }

        logger.info("教授处理器初始化完成")

    # --------------------------------------------------------
    # 单教授处理
    # --------------------------------------------------------

    def process(
        self,
        professor_folder: Path,
        max_papers: Optional[int] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        完整处理一位教授。

        Args:
            professor_folder: 教授文件夹路径
            max_papers: 最多处理几篇论文（覆盖默认值）
            force: 是否强制重新处理（忽略已有摘要缓存）

        Returns:
            {
                "professor_name": str,
                "professor_folder": str,
                "status": "completed" | "partial" | "failed",
                "steps": {
                    "load_info": {"status": ..., "elapsed": ...},
                    "download_papers": {"status": ..., "elapsed": ..., "result": ...},
                    "read_papers": {"status": ..., "elapsed": ..., "result": ...},
                    "update_status": {"status": ..., "elapsed": ...},
                },
                "total_elapsed_seconds": float,
            }
        """
        prof_dir = Path(professor_folder)
        info_path = prof_dir / "info.json"

        max_p = max_papers or self.max_papers_per_professor

        result = {
            "professor_name": prof_dir.name,
            "professor_folder": str(prof_dir),
            "status": "pending",
            "steps": {},
            "total_elapsed_seconds": 0,
        }

        t_start = time.time()
        professor_info = {}

        # ── Step 1: 加载信息 ──
        t0 = time.time()
        try:
            if not info_path.exists():
                result["status"] = "failed"
                result["steps"]["load_info"] = {
                    "status": "failed",
                    "error": f"info.json 不存在: {info_path}",
                    "elapsed": time.time() - t0,
                }
                logger.error(f"教授文件夹缺少 info.json: {prof_dir}")
                return result

            professor_info = load_json(str(info_path))
            name = professor_info.get("name", prof_dir.name)
            logger.info(f"开始处理教授: {name}")

            result["steps"]["load_info"] = {
                "status": "success",
                "elapsed": round(time.time() - t0, 2),
            }
        except Exception as e:
            logger.error(f"加载 info.json 失败: {e}")
            result["status"] = "failed"
            result["steps"]["load_info"] = {
                "status": "failed",
                "error": str(e),
                "elapsed": round(time.time() - t0, 2),
            }
            return result

        # ── Step 2: 下载论文 ──
        self._update_professor_status(prof_dir, "papers_downloading",
                                       "开始下载论文")
        t1 = time.time()
        try:
            download_result = self.downloader.download_papers(
                professor_info,
                max_papers=max_p,
                prof_dir=str(prof_dir),
            )

            n_downloaded = len(download_result.get("downloaded", []))
            n_cached = len(download_result.get("cached", []))
            n_failed = len(download_result.get("failed", []))
            n_total = n_downloaded + n_cached

            self._stats["total_papers_downloaded"] += n_total

            result["steps"]["download_papers"] = {
                "status": "success" if n_total > 0 else "partial",
                "downloaded": n_downloaded,
                "cached": n_cached,
                "failed": n_failed,
                "elapsed": round(time.time() - t1, 2),
            }

            logger.info(
                f"{name} 论文下载: {n_total} 成功, {n_failed} 失败"
            )

        except Exception as e:
            logger.error(f"论文下载失败: {e}")
            result["steps"]["download_papers"] = {
                "status": "failed",
                "error": str(e),
                "elapsed": round(time.time() - t1, 2),
            }

        # ── Step 3: 生成摘要 ──
        self._update_professor_status(prof_dir, "papers_reading",
                                       "开始生成论文摘要")
        t2 = time.time()
        try:
            read_results = self.reader.batch_read(
                professor_info,
                max_papers=max_p,
                prof_dir=str(prof_dir),
            )

            n_success = sum(1 for r in read_results if r["read_status"] == "success")
            self._stats["total_papers_read"] += n_success

            result["steps"]["read_papers"] = {
                "status": "success" if n_success > 0 else "partial",
                "success": n_success,
                "total": len(read_results),
                "elapsed": round(time.time() - t2, 2),
            }

            logger.info(
                f"{name} 论文阅读: {n_success}/{len(read_results)} 成功"
            )

        except Exception as e:
            logger.error(f"论文阅读失败: {e}")
            result["steps"]["read_papers"] = {
                "status": "failed",
                "error": str(e),
                "elapsed": round(time.time() - t2, 2),
            }

        # ── Step 4: 更新状态 ──
        t3 = time.time()
        try:
            self._update_professor_status(
                prof_dir,
                "paper_reading_completed",
                f"论文处理完成: "
                f"下载 {result['steps'].get('download_papers', {}).get('downloaded', 0) + result['steps'].get('download_papers', {}).get('cached', 0)} 篇, "
                f"阅读 {result['steps'].get('read_papers', {}).get('success', 0)} 篇",
            )

            # 更新 info.json 中的状态
            professor_info["status"] = "paper_reading_completed"
            professor_info["paper_reading_completed_at"] = datetime.now().isoformat()
            save_json(professor_info, str(info_path))

            result["steps"]["update_status"] = {
                "status": "success",
                "elapsed": round(time.time() - t3, 2),
            }
        except Exception as e:
            logger.error(f"状态更新失败: {e}")
            result["steps"]["update_status"] = {
                "status": "failed",
                "error": str(e),
                "elapsed": round(time.time() - t3, 2),
            }

        # ── 汇总 ──
        total_elapsed = time.time() - t_start
        result["total_elapsed_seconds"] = round(total_elapsed, 1)
        self._stats["total_time_seconds"] += total_elapsed

        # 判断最终状态
        load_ok = result["steps"].get("load_info", {}).get("status") == "success"
        dl_ok = result["steps"].get("download_papers", {}).get("status") in ("success", "partial")
        read_ok = result["steps"].get("read_papers", {}).get("status") in ("success", "partial")

        if load_ok and dl_ok and read_ok:
            result["status"] = "completed"
            self._stats["total_success"] += 1
        elif load_ok and (dl_ok or read_ok):
            result["status"] = "partial"
            self._stats["total_success"] += 1
        else:
            result["status"] = "failed"
            self._stats["total_failed"] += 1

        self._stats["total_processed"] += 1

        logger.info(
            f"{name} 处理{result['status']} "
            f"({total_elapsed:.1f}s)"
        )

        return result

    # --------------------------------------------------------
    # 批量处理
    # --------------------------------------------------------

    def batch_process(
        self,
        professor_folders: List[Path],
        max_papers: Optional[int] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        批量处理多位教授。

        Args:
            professor_folders: 教授文件夹路径列表
            max_papers: 每位教授最多处理几篇论文
            force: 是否强制重新处理

        Returns:
            {
                "total": int,
                "completed": int,
                "partial": int,
                "failed": int,
                "total_elapsed": float,
                "results": [单个结果, ...],
            }
        """
        total = len(professor_folders)
        logger.info(f"开始批量处理 {total} 位教授")

        all_results = []
        t_start = time.time()

        with tqdm(total=total, desc="处理教授", unit="prof", ncols=80) as pbar:
            for prof_dir in professor_folders:
                pbar.set_postfix_str(f"{prof_dir.name[:30]}")

                try:
                    result = self.process(prof_dir, max_papers=max_papers, force=force)
                    all_results.append(result)
                except Exception as e:
                    logger.error(f"教授 {prof_dir.name} 处理异常: {e}")
                    all_results.append({
                        "professor_name": prof_dir.name,
                        "professor_folder": str(prof_dir),
                        "status": "failed",
                        "error": str(e),
                    })

                pbar.update(1)

                # 更新进度条描述
                completed = sum(1 for r in all_results if r["status"] == "completed")
                failed = sum(1 for r in all_results if r["status"] == "failed")
                pbar.set_description(f"处理 ({completed}✓ {failed}✗)")

        total_elapsed = round(time.time() - t_start, 1)

        summary = {
            "total": total,
            "completed": sum(1 for r in all_results if r["status"] == "completed"),
            "partial": sum(1 for r in all_results if r["status"] == "partial"),
            "failed": sum(1 for r in all_results if r["status"] == "failed"),
            "total_elapsed_seconds": total_elapsed,
            "results": all_results,
        }

        logger.info(
            f"批量处理完成: {summary['completed']}成功 {summary['partial']}部分 "
            f"{summary['failed']}失败 ({total_elapsed:.1f}s)"
        )

        return summary

    # --------------------------------------------------------
    # 辅助方法
    # --------------------------------------------------------

    def _update_professor_status(
        self, prof_dir: Path, new_status: str, note: str = ""
    ) -> None:
        """更新教授状态文件"""
        status_path = prof_dir / "status.json"

        status = {}
        if status_path.exists():
            try:
                status = load_json(str(status_path))
            except Exception:
                pass

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

    def get_stats(self) -> Dict[str, Any]:
        """返回处理统计"""
        return dict(self._stats)

    def reset_stats(self) -> None:
        """重置统计"""
        for key in self._stats:
            if isinstance(self._stats[key], (int, float)):
                self._stats[key] = 0

    def print_summary(self, batch_result: Dict[str, Any]) -> None:
        """打印批量处理摘要"""
        print(f"\n{'═' * 60}")
        print(f"  教授处理摘要")
        print(f"{'─' * 60}")
        print(f"  总数:       {batch_result.get('total', 0)}")
        print(f"  完成:       {batch_result.get('completed', 0)} ✅")
        print(f"  部分:       {batch_result.get('partial', 0)} ⚠️")
        print(f"  失败:       {batch_result.get('failed', 0)} ❌")
        print(f"  总耗时:     {batch_result.get('total_elapsed_seconds', 0):.1f}s")
        print(f"{'─' * 60}")

        # 列出失败项
        failures = [r for r in batch_result.get("results", []) if r["status"] == "failed"]
        if failures:
            print(f"\n  失败详情:")
            for f in failures:
                err = f.get("steps", {}).get("load_info", {}).get("error",
                       f.get("steps", {}).get("download_papers", {}).get("error",
                       f.get("steps", {}).get("read_papers", {}).get("error",
                       f.get("error", "Unknown"))))
                print(f"    ❌ {f['professor_name'][:35]}: {str(err)[:60]}")

        # 列出部分完成项
        partials = [r for r in batch_result.get("results", []) if r["status"] == "partial"]
        if partials:
            print(f"\n  部分完成:")
            for p in partials:
                issues = []
                steps = p.get("steps", {})
                if steps.get("download_papers", {}).get("failed", 0) > 0:
                    issues.append(f"下载失败 {steps['download_papers']['failed']}")
                if steps.get("read_papers", {}).get("status") == "failed":
                    issues.append("阅读失败")
                print(f"    ⚠️ {p['professor_name'][:35]}: {', '.join(issues)}")

        print(f"{'═' * 60}\n")


# ============================================================
# 自测入口
# ============================================================

if __name__ == "__main__":
    from scripts.profile_parser import ProfileParser
    from scripts.llm_client import LLMClient
    from scripts.paper_cache import PaperCache

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    print(f"\n{'═' * 60}")
    print("  教授处理器 - 自测")
    print(f"{'═' * 60}")

    # 初始化组件
    parser = ProfileParser("profiles/my_profile_template.json")
    client = LLMClient()
    cache = PaperCache()
    downloader = PaperDownloader(cache)
    reader = PaperReader(parser, client, cache)
    processor = ProfessorProcessor(downloader, reader)

    # 查找现有的教授文件夹
    prof_dirs = sorted(Path("professors").glob("*"))
    prof_dirs = [d for d in prof_dirs if d.is_dir() and (d / "info.json").exists()]

    if prof_dirs:
        print(f"\n找到 {len(prof_dirs)} 个教授文件夹")

        # 处理第一个教授
        print(f"\n[1] 处理教授: {prof_dirs[0].name}")
        result = processor.process(prof_dirs[0], max_papers=1)

        print(f"\n  结果:")
        print(f"    状态: {result['status']}")
        print(f"    耗时: {result['total_elapsed_seconds']}s")
        for step_name, step_info in result.get("steps", {}).items():
            status = step_info.get("status", "?")
            elapsed = step_info.get("elapsed", "?")
            icon = {"success": "✅", "partial": "⚠️", "failed": "❌"}.get(status, "❓")
            print(f"    {icon} {step_name}: {status} ({elapsed}s)")

        # 打印统计
        print(f"\n[2] 处理统计:")
        stats = processor.get_stats()
        for k, v in stats.items():
            print(f"    {k}: {v}")

        # 检查处理后的文件夹
        print(f"\n[3] 文件夹内容:")
        for f in sorted((prof_dirs[0] / "papers").glob("*")):
            print(f"    {f.name}")
    else:
        print("\n⚠️ 没有找到教授文件夹。")
        print("   请先运行搜索流程，或手动创建测试数据。")

        # 创建测试数据
        print("\n[测试] 创建模拟教授文件夹...")
        test_dir = Path("professors/Test_University_Dr._Test")
        ensure_directory(str(test_dir))
        ensure_directory(str(test_dir / "papers"))

        save_json({
            "professor_id": "test_001",
            "name": "Dr. Test",
            "institution": "Test University",
            "research_topics": ["deep learning", "protein structure"],
            "publication_count": 10,
            "h_index": 12,
            "recent_papers": [{
                "title": "Attention Is All You Need",
                "year": 2017,
                "citations": 100000,
                "paper_id": "",
                "external_ids": {"ArXiv": "1706.03762"},
            }],
            "status": "pending_score",
            "search_round": "test",
        }, str(test_dir / "info.json"))

        save_json({
            "current_status": "pending_score",
            "status_history": [{"status": "pending_score", "timestamp": datetime.now().isoformat(), "note": "测试数据"}],
            "last_updated": datetime.now().isoformat(),
        }, str(test_dir / "status.json"))

        print("    ✅ 测试数据已创建")

        # 处理
        result = processor.process(test_dir, max_papers=1)
        print(f"\n    处理结果: {result['status']} ({result['total_elapsed_seconds']}s)")

    print(f"\n✅ 自测完成")
