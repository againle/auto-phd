"""
自适应 Agent 引擎 — 支持全自动/半自动两种模式

全自动模式：目标驱动 + 三种自适应策略 + 智能停止
"""

import logging, threading, time, json
from datetime import datetime
from typing import Dict, List, Optional, Callable

from src.state_manager import GlobalState, ProfessorStatus, SearchStrategy
from src.data_manager import DataManager
from scripts.llm_client import LLMClient, TaskType
from scripts.profile_parser import ProfileParser
from scripts.scholar_api import search_professors
from scripts.professor_scorer import ProfessorScorer
from scripts.paper_downloader import PaperDownloader
from scripts.paper_reader import PaperReader
from scripts.paper_cache import PaperCache
from scripts.email_generator import EmailGenerator
from scripts.utils import load_config

logger = logging.getLogger(__name__)

# ── 自适应策略 Prompt ──

STRATEGY_ANALYSIS_PROMPT = """Analyze the current search results and recommend next-round strategy.

Current State:
- Target: {target} qualified professors
- Found so far: {total_found} total, {qualified} qualified
- Current round: {round_num}, Strategy: {current_strategy}
- Keywords used: {used_keywords}
- Regions used: {used_regions}
- Consecutive stagnant rounds: {stagnant}

Best performing keyword: {best_keyword} ({best_count} professors)
Best performing region: {best_region}

My research interests: {my_interests}

Recommend next-round parameters. Return ONLY JSON:
{{
    "strategy": "carpet_bomb" | "precision" | "snowball",
    "keywords": ["kw1", "kw2", "kw3"],
    "regions": ["region1", "region2"],
    "max_results": 20,
    "reasoning": "one sentence explaining the choice"
}}

Rules:
- carpet_bomb: use when <10 qualified, broad keywords for wide coverage
- precision: use when 10-25 qualified, focus on best-performing directions  
- snowball: use when >25 qualified or stagnant, trace citation networks
- NEVER repeat already-used keywords exactly
- If a region performed poorly (>50% low-score), replace it
- If stagnant for 2+ rounds, broaden keywords significantly"""


class AgentEngine:
    def __init__(self, state: GlobalState):
        self.state = state
        self.data_mgr = DataManager()
        self.is_running = False
        self._stop_flag = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._llm: Optional[LLMClient] = None
        self._profile: Optional[ProfileParser] = None
        self._scorer: Optional[ProfessorScorer] = None
        self._cache: Optional[PaperCache] = None
        self._downloader: Optional[PaperDownloader] = None
        self._reader: Optional[PaperReader] = None
        self._generator: Optional[EmailGenerator] = None

        self._callbacks: Dict[str, List[Callable]] = {
            "on_professor_found": [], "on_round_complete": [],
            "on_processing_complete": [], "on_error": [], "on_progress": [],
            "on_stop": [], "on_strategy_change": [],
        }
        self._progress = {"current": 0, "total": 0, "message": ""}

    # ── 懒加载 ──
    @property
    def llm(self): 
        if self._llm is None: self._llm = LLMClient()
        return self._llm
    @property
    def profile(self):
        if self._profile is None: self._profile = ProfileParser("profiles/my_profile.json")
        return self._profile
    @property
    def scorer(self):
        if self._scorer is None: self._scorer = ProfessorScorer(self.profile)
        return self._scorer
    @property
    def cache(self):
        if self._cache is None: self._cache = PaperCache("data/cache")
        return self._cache
    @property
    def downloader(self):
        if self._downloader is None: self._downloader = PaperDownloader(self.cache)
        return self._downloader
    @property
    def reader(self):
        if self._reader is None: self._reader = PaperReader(self.profile, self.llm, self.cache)
        return self._reader
    @property
    def generator(self):
        if self._generator is None: self._generator = EmailGenerator(self.profile, self.llm)
        return self._generator

    # ── 回调 ──
    def on(self, event: str, callback: Callable):
        if event in self._callbacks: self._callbacks[event].append(callback)
    def _emit(self, event: str, **data):
        for cb in self._callbacks.get(event, []):
            try: cb(**data)
            except Exception as e: logger.error(f"Callback {event}: {e}")

    # ═══════════════════════════════════════
    # 全自动主循环
    # ═══════════════════════════════════════

    def run_auto_loop(self):
        """全自动模式主循环（在线程中运行）"""
        self.is_running = True
        self._stop_flag.clear()
        config = load_config()
        auto = config.get("auto_mode", {})

        # 从 config 同步参数
        self.state.auto_target = auto.get("target_professors", self.state.auto_target)
        self.state.auto_min_score = auto.get("min_quality_score", self.state.auto_min_score)
        self.state.auto_max_rounds = auto.get("max_rounds", self.state.auto_max_rounds)
        self.state.auto_stagnant_stop = auto.get("stop_after_stagnant_rounds", self.state.auto_stagnant_stop)
        self.state.auto_keyword_limit = auto.get("stop_after_keyword_exhaustion", self.state.auto_keyword_limit)
        self.state.auto_max_cost = auto.get("max_cost_usd", self.state.auto_max_cost)

        self.state.add_strategy_log(f"Auto mode started. Target: {self.state.auto_target} professors, min score: {self.state.auto_min_score}")

        while not self._stop_flag.is_set():
            self.state.current_round += 1
            round_num = self.state.current_round

            # ── 检查停止条件 ──
            should_stop, reason = self.state.should_stop_auto()
            if should_stop:
                self.state.add_strategy_log(f"STOP: {reason}")
                self._emit("on_stop", reason=reason)
                break

            # ── 制定本轮策略 ──
            self._set_progress(0, 1, f"Round {round_num}: planning strategy...")
            plan = self._plan_next_round()

            self.state.add_strategy_log(
                f"Round {round_num}: {plan['strategy']} | {plan.get('keywords',[])[:3]} | {plan.get('regions',[])}"
            )
            self._emit("on_strategy_change", plan=plan, round_num=round_num)

            # ── 执行搜索 ──
            prev_total = self.state.total_found
            keywords = plan.get("keywords", [])
            year_range = plan.get("time_range_years", 3)
            max_per = plan.get("max_results", 20)

            total_kw = len(keywords)
            for i, kw in enumerate(keywords):
                if self._stop_flag.is_set(): break
                self._set_progress(i + 1, total_kw, f"Searching: {kw}")
                if kw not in self.state.used_keywords:
                    self.state.used_keywords.append(kw)
                try:
                    results = search_professors(kw, limit=min(max_per, 30), year=year_range)
                    for prof in results:
                        sid = prof.get("scholar_id", "")
                        if sid and sid not in self.state.professors:
                            sr = self.scorer.score(prof)
                            prof["_score"] = sr["total_score"]
                            prof["_breakdown"] = sr["breakdown"]
                            pid = self.state.add_professor(prof)
                            self.data_mgr.save_professor(pid, prof)
                            if prof["_score"] >= self.state.auto_min_score:
                                self.state.select_professor(pid)
                            self._emit("on_professor_found", professor_id=pid, professor=prof)
                except Exception as e:
                    logger.error(f"Search '{kw}' failed: {e}")
                if i < total_kw - 1: time.sleep(1.5)

            # ── 更新停滞计数 ──
            new_count = self.state.total_found - prev_total
            if new_count == 0:
                self.state.consecutive_no_new += 1
            else:
                self.state.consecutive_no_new = 0

            # ── 记录本轮 ──
            self.state.round_history.append({
                "round": round_num,
                "timestamp": datetime.now().isoformat(),
                "strategy": plan.get("strategy", ""),
                "keywords": keywords,
                "regions": plan.get("regions", []),
                "found": new_count,
                "total_qualified": self.state.total_qualified,
                "notes": plan.get("reasoning", ""),
            })

            self.state.save()
            self._emit("on_round_complete", round_num=round_num, new_found=new_count)

            # 轮间休息
            sleep_sec = config.get("circuit_breaker", {}).get("sleep_between_rounds", 10)
            for _ in range(min(sleep_sec, 30)):
                if self._stop_flag.is_set(): break
                time.sleep(1)

        self.is_running = False
        self.state.add_strategy_log(f"Auto mode ended. Qualified: {self.state.total_qualified}/{self.state.auto_target}")
        self.state.save()

    # ═══════════════════════════════════════
    # 自适应策略规划
    # ═══════════════════════════════════════

    def _plan_next_round(self) -> Dict:
        """根据当前状态决定下一轮策略"""
        q = self.state.total_qualified
        target = self.state.auto_target
        round_num = self.state.current_round

        # 启发式策略选择
        if q < 10 and round_num <= 3:
            return self._carpet_bomb()
        elif q >= 25 or self.state.consecutive_no_new >= 2:
            return self._snowball()
        else:
            # 用 LLM 做精准策略
            try:
                return self._llm_strategy()
            except Exception:
                return self._precision_strike()

    def _carpet_bomb(self) -> Dict:
        """地毯式轰炸：宽泛关键词，广撒网"""
        prefs = self.profile.profile.target_preferences
        keywords = [kw for kw in self.profile.get_research_keywords()[:8]
                    if kw not in self.state.used_keywords]
        if not keywords:
            # 用 LLM 扩展
            try:
                ext = self._expand_keywords()
                keywords = [k for k in ext if k not in self.state.used_keywords][:5]
            except Exception:
                keywords = self.profile.get_research_keywords()[:5]

        regions = prefs.locations[:] if prefs.locations else ["United States", "Canada"]
        self.state.current_strategy = SearchStrategy.CARPET_BOMB.value

        return {
            "strategy": "carpet_bomb",
            "keywords": keywords or ["machine learning"],
            "regions": regions,
            "time_range_years": 3,
            "max_results": 25,
            "reasoning": f"Carpet bombing round {self.state.current_round} to build base coverage",
        }

    def _precision_strike(self) -> Dict:
        """精准打击：聚焦表现最好的方向"""
        # 找最佳关键词（产生最多合格教授的）
        kw_perf = {}
        for pid in self.state.selected_ids:
            prof = self.state.professors.get(pid, {})
            for topic in prof.get("research_topics", [])[:3]:
                kw_perf[topic] = kw_perf.get(topic, 0) + 1

        best_kw = sorted(kw_perf.items(), key=lambda x: x[1], reverse=True)
        keywords = [k for k, _ in best_kw[:5] if k not in self.state.used_keywords]

        if not keywords:
            keywords = [k for k, _ in best_kw[:5]]

        regions = self.state.used_regions[:3] if self.state.used_regions else ["United States"]
        self.state.current_strategy = SearchStrategy.PRECISION_STRIKE.value

        return {
            "strategy": "precision",
            "keywords": keywords or ["deep learning"],
            "regions": regions,
            "time_range_years": 3,
            "max_results": 15,
            "reasoning": f"Precision strike on best-performing directions: {keywords[:3]}",
        }

    def _snowball(self) -> Dict:
        """雪球效应：扩展关键词 + 新地区"""
        try:
            new_kw = self._expand_keywords()
            keywords = [k for k in new_kw if k not in self.state.used_keywords][:5]
        except Exception:
            keywords = self.profile.get_research_keywords()[:5]

        # 扩展地区
        all_regions = ["United States", "Canada", "United Kingdom", "Switzerland",
                       "Germany", "Australia", "Singapore", "Netherlands"]
        new_regions = [r for r in all_regions if r not in self.state.used_regions][:3]
        if not new_regions:
            new_regions = ["United States", "Canada"]

        self.state.current_strategy = SearchStrategy.SNOWBALL.value
        return {
            "strategy": "snowball",
            "keywords": keywords or ["machine learning AI"],
            "regions": new_regions,
            "time_range_years": 3,
            "max_results": 20,
            "reasoning": f"Snowball: expanding to {new_regions} with {keywords[:3]}",
        }

    def _llm_strategy(self) -> Dict:
        """用 LLM 分析并推荐策略"""
        # 找最佳关键词和地区
        kw_count = {}
        region_count = {}
        for pid in self.state.selected_ids:
            prof = self.state.professors.get(pid, {})
            for t in prof.get("research_topics", [])[:2]:
                kw_count[t] = kw_count.get(t, 0) + 1
            inst = prof.get("institution", "")
            for r in self.state.used_regions:
                if r.lower() in inst.lower():
                    region_count[r] = region_count.get(r, 0) + 1

        best_kw = max(kw_count, key=kw_count.get) if kw_count else "N/A"
        best_region = max(region_count, key=region_count.get) if region_count else "N/A"

        prompt = STRATEGY_ANALYSIS_PROMPT.format(
            target=self.state.auto_target,
            total_found=self.state.total_found,
            qualified=self.state.total_qualified,
            round_num=self.state.current_round,
            current_strategy=self.state.current_strategy,
            used_keywords=", ".join(self.state.used_keywords[-10:]),
            used_regions=", ".join(self.state.used_regions[-5:]),
            stagnant=self.state.consecutive_no_new,
            best_keyword=best_kw,
            best_count=kw_count.get(best_kw, 0),
            best_region=best_region,
            my_interests=", ".join(self.profile.get_research_keywords()[:8]),
        )

        resp = self.llm.call(messages=[{"role": "user", "content": prompt}], task_type=TaskType.GENERAL)
        try:
            cleaned = resp.strip()
            if cleaned.startswith("```"): cleaned = "\n".join(cleaned.split("\n")[1:-1])
            plan = json.loads(cleaned)
            plan["time_range_years"] = 3
            self.state.current_strategy = plan.get("strategy", SearchStrategy.PRECISION_STRIKE.value)
            return plan
        except Exception:
            return self._precision_strike()

    def _expand_keywords(self) -> List[str]:
        """用 LLM 扩展关键词"""
        prompt = f"""My research keywords: {', '.join(self.profile.get_research_keywords())}
Already used: {', '.join(self.state.used_keywords[-10:])}
Suggest 5 NEW search keywords for finding professors. Return as JSON list: ["kw1","kw2",...]"""
        resp = self.llm.call(messages=[{"role": "user", "content": prompt}], task_type=TaskType.GENERAL)
        try:
            cleaned = resp.strip()
            if cleaned.startswith("```"): cleaned = "\n".join(cleaned.split("\n")[1:-1])
            return json.loads(cleaned)
        except Exception:
            return self.profile.get_research_keywords()[:5]

    # ═══════════════════════════════════════
    # 半自动：单轮搜索
    # ═══════════════════════════════════════

    def start_search(self, plan: Dict) -> Dict:
        self.is_running = True; self._stop_flag.clear()
        self.state.current_round += 1
        keywords = plan.get("keywords", [])
        for kw in keywords:
            if kw not in self.state.used_keywords: self.state.used_keywords.append(kw)
        found = 0
        for kw in keywords:
            if self._stop_flag.is_set(): break
            try:
                results = search_professors(kw, limit=min(plan.get("max_results_per_query", 20), 30), year=plan.get("time_range_years", 3))
                for prof in results:
                    sid = prof.get("scholar_id", "")
                    if sid and sid not in self.state.professors:
                        sr = self.scorer.score(prof)
                        prof["_score"] = sr["total_score"]
                        prof["_breakdown"] = sr["breakdown"]
                        pid = self.state.add_professor(prof)
                        self.data_mgr.save_professor(pid, prof)
                        found += 1
            except Exception as e:
                logger.error(f"Search '{kw}' failed: {e}")
        self.state.pending_ids = [pid for pid, p in self.state.professors.items()
                                   if p.get("_status") == ProfessorStatus.SEARCHED.value]
        self.state.current_plan = plan
        self.state.save()
        self.is_running = False
        return {"round_id": f"round_{self.state.current_round:03d}", "professors_found": found}

    # ═══════════════════════════════════════
    # 处理选中教授 + 进度
    # ═══════════════════════════════════════

    def continue_processing(self, selected_ids: List[str]) -> Dict:
        self.is_running = True; self._stop_flag.clear()
        processed = failed = generated = 0
        for pid in selected_ids:
            if self._stop_flag.is_set(): break
            prof = self.state.professors.get(pid)
            if not prof: continue
            self.state.update_status(pid, ProfessorStatus.PROCESSING)
            try:
                dl = self.downloader.download_papers(prof, max_papers=2)
                if prof.get("recent_papers"):
                    rr = self.reader.batch_read(prof, max_papers=2)
                    if any(r["read_status"] == "success" for r in rr):
                        self.state.update_status(pid, ProfessorStatus.PAPER_READ)
                        try:
                            from pathlib import Path
                            pd = Path("professors") / pid
                            result = self.generator.generate_emails(pd, styles=["academic"])
                            if result.get("versions"):
                                self.state.update_status(pid, ProfessorStatus.EMAIL_GENERATED)
                                self.state.total_emails_generated += 1
                                generated += 1
                        except Exception: pass
                processed += 1
            except Exception as e:
                logger.error(f"Process {pid} failed: {e}"); failed += 1
            self.state.save()
        self.state.save(); self.is_running = False
        self._emit("on_processing_complete", processed=processed, failed=failed, generated=generated)
        return {"processed": processed, "failed": failed, "email_generated": generated}

    def get_search_progress(self) -> Dict:
        return {**self.state.get_stats(), "step_progress": self._progress, "is_running": self.is_running}

    def _set_progress(self, current, total, message):
        self._progress = {"current": current, "total": total, "message": message}
        self._emit("on_progress", current=current, total=total, message=message)

    def stop(self):
        self._stop_flag.set(); self.is_running = False
        logger.info("Engine stopped")
