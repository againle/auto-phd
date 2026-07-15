"""
数据管理器 — 教授数据 JSON 持久化
"""

import json, logging
from pathlib import Path
from typing import Dict, List, Optional
from src.state_manager import GlobalState
from src import ensure_directories

logger = logging.getLogger(__name__)


class DataManager:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.prof_dir = self.data_dir / "professors"
        ensure_directories()

    def save_professor(self, prof_id: str, data: Dict):
        path = self.prof_dir / f"{prof_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    def load_professor(self, prof_id: str) -> Optional[Dict]:
        path = self.prof_dir / f"{prof_id}.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def delete_professor(self, prof_id: str):
        path = self.prof_dir / f"{prof_id}.json"
        if path.exists(): path.unlink()

    def get_all_professors(self) -> List[Dict]:
        results = []
        for f in sorted(self.prof_dir.glob("*.json")):
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    results.append(json.load(fh))
            except Exception as e:
                logger.warning(f"Load {f.name} failed: {e}")
        return results

    def save_state(self, state: GlobalState):
        state.save(str(self.data_dir / "state.json"))

    def load_state(self) -> GlobalState:
        return GlobalState.load(str(self.data_dir / "state.json"))
