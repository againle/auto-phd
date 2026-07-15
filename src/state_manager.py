"""
全局状态管理器 — 支持多轮搜索累加 + 全自动模式
"""

import json, logging
from enum import Enum
from dataclasses import dataclass, field, asdict, MISSING
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class ProfessorStatus(Enum):
    PENDING="pending"; SEARCHED="searched"; SCORED="scored"
    SELECTED="selected"; REJECTED="rejected"; PROCESSING="processing"
    PAPER_READ="paper_read"; EMAIL_GENERATED="email_generated"
    EMAIL_READY="email_ready"; EMAIL_SENT="email_sent"


class SearchStrategy(Enum):
    CARPET_BOMB="carpet_bomb"
    PRECISION_STRIKE="precision"
    SNOWBALL="snowball"


@dataclass
class GlobalState:
    run_mode: str = "semi_auto"
    auto_target: int = 30
    auto_min_score: int = 50
    auto_max_rounds: int = 10
    auto_stagnant_stop: int = 3
    auto_keyword_limit: int = 20
    auto_max_cost: float = 0.50
    current_round: int = 0
    is_running: bool = False
    current_strategy: str = SearchStrategy.CARPET_BOMB.value
    used_keywords: List[str] = field(default_factory=list)
    used_regions: List[str] = field(default_factory=list)
    consecutive_no_new: int = 0
    total_api_cost: float = 0.0
    professors: Dict[str, Dict] = field(default_factory=dict)
    selected_ids: List[str] = field(default_factory=list)
    rejected_ids: List[str] = field(default_factory=list)
    round_history: List[Dict] = field(default_factory=list)
    current_plan: Optional[Dict] = None
    pending_ids: List[str] = field(default_factory=list)
    total_found: int = 0
    total_qualified: int = 0
    total_selected: int = 0
    total_emails_generated: int = 0
    total_emails_sent: int = 0
    strategy_log: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        d = {}
        for f in self.__dataclass_fields__:
            v = getattr(self, f)
            if isinstance(v, Enum): v = v.value
            d[f] = v
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> "GlobalState":
        merged = {}
        for f in cls.__dataclass_fields__.values():
            key = f.name
            if key in data:
                merged[key] = data[key]
            elif f.default_factory is not MISSING:
                merged[key] = f.default_factory()
            elif f.default is not MISSING:
                merged[key] = f.default
            else:
                merged[key] = None
        return cls(**merged)

    def save(self, path: str = "data/state.json"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2, default=str)

    @classmethod
    def load(cls, path: str = "data/state.json") -> "GlobalState":
        p = Path(path)
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return cls.from_dict(json.load(f))
            except Exception as e:
                logger.warning(f"Load state failed: {e}")
        return cls()

    # ── Professor CRUD ──
    def add_professor(self, prof_data: Dict) -> str:
        sid = prof_data.get("scholar_id", f"prof_{len(self.professors):04d}")
        if sid not in self.professors:
            prof_data["_status"] = ProfessorStatus.SEARCHED.value
            prof_data["_added_at"] = datetime.now().isoformat()
            self.professors[sid] = prof_data
            self.total_found += 1
        return sid

    def select_professor(self, prof_id: str):
        if prof_id in self.professors and prof_id not in self.selected_ids:
            self.professors[prof_id]["_status"] = ProfessorStatus.SELECTED.value
            self.selected_ids.append(prof_id); self.total_selected += 1
            if prof_id in self.rejected_ids: self.rejected_ids.remove(prof_id)

    def reject_professor(self, prof_id: str):
        if prof_id in self.professors:
            self.professors[prof_id]["_status"] = ProfessorStatus.REJECTED.value
            if prof_id not in self.rejected_ids: self.rejected_ids.append(prof_id)
            if prof_id in self.selected_ids:
                self.selected_ids.remove(prof_id); self.total_selected = max(0, self.total_selected - 1)

    def update_status(self, prof_id: str, status: ProfessorStatus):
        if prof_id in self.professors:
            self.professors[prof_id]["_status"] = status.value

    def get_pending_professors(self) -> List[Dict]:
        return [p for pid, p in self.professors.items()
                if p.get("_status") in (ProfessorStatus.SEARCHED.value, ProfessorStatus.SCORED.value)]

    def get_selected_professors(self) -> List[Dict]:
        return [self.professors[pid] for pid in self.selected_ids if pid in self.professors]

    def add_strategy_log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.strategy_log.append(f"[{ts}] {msg}")
        if len(self.strategy_log) > 80: self.strategy_log = self.strategy_log[-80:]

    def should_stop_auto(self) -> tuple:
        q = sum(1 for pid in self.selected_ids
                if self.professors.get(pid, {}).get("_score", 0) >= self.auto_min_score)
        self.total_qualified = q
        if q >= self.auto_target: return True, f"Target reached: {q}/{self.auto_target}"
        if self.consecutive_no_new >= self.auto_stagnant_stop:
            return True, f"Stagnation: {self.auto_stagnant_stop} rounds no new"
        if len(self.used_keywords) >= self.auto_keyword_limit:
            return True, f"Keyword exhaustion: {len(self.used_keywords)}"
        if self.total_api_cost >= self.auto_max_cost:
            return True, f"Cost limit: ${self.total_api_cost:.4f}"
        if self.current_round >= self.auto_max_rounds:
            return True, f"Max rounds: {self.current_round}"
        return False, None

    def get_stats(self) -> Dict:
        q = sum(1 for pid in self.selected_ids
                if self.professors.get(pid, {}).get("_score", 0) >= self.auto_min_score)
        return {
            "total_found": self.total_found, "total_qualified": q,
            "total_selected": self.total_selected, "total_rejected": len(self.rejected_ids),
            "pending": len(self.get_pending_professors()),
            "emails_generated": self.total_emails_generated,
            "emails_sent": self.total_emails_sent,
            "current_round": self.current_round, "target": self.auto_target,
            "strategy": self.current_strategy, "consecutive_no_new": self.consecutive_no_new,
            "api_cost": round(self.total_api_cost, 4),
        }
