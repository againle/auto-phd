"""
集成测试 — 验证完整流程
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.state_manager import GlobalState, ProfessorStatus
from src.data_manager import DataManager
from src.agent_engine import AgentEngine


def test_state_manager():
    """测试状态管理器"""
    print("[1] 测试状态管理器...")
    state = GlobalState()
    state.add_professor({
        "name": "Test Professor",
        "institution": "Test University",
        "scholar_id": "test_001",
        "research_topics": ["AI", "ML"],
        "h_index": 15,
        "publication_count": 20,
        "recent_papers": [{"title": "Test Paper", "year": 2025, "citations": 10}],
    })

    assert state.total_found == 1
    assert len(state.get_pending_professors()) == 1

    state.select_professor("test_001")
    assert state.total_selected == 1
    assert len(state.get_selected_professors()) == 1

    state.reject_professor("test_001")
    assert len(state.rejected_ids) == 1
    assert state.total_selected == 0

    state.save("data/test_state.json")
    loaded = GlobalState.load("data/test_state.json")
    assert loaded.total_found == 1
    Path("data/test_state.json").unlink()

    print("   ✅ 状态管理器测试通过")


def test_data_manager():
    """测试数据管理器"""
    print("[2] 测试数据管理器...")
    dm = DataManager()
    dm.save_professor("test_001", {"name": "Test", "scholar_id": "test_001"})
    loaded = dm.load_professor("test_001")
    assert loaded["name"] == "Test"
    dm.delete_professor("test_001")
    assert dm.load_professor("test_001") is None
    print("   ✅ 数据管理器测试通过")


def test_agent_engine_init():
    """测试Agent引擎初始化"""
    print("[3] 测试Agent引擎...")
    state = GlobalState()
    engine = AgentEngine(state)
    assert not engine.is_running
    progress = engine.get_search_progress()
    assert progress["total_found"] == 0
    assert progress["target"] == 30
    print("   ✅ Agent引擎初始化测试通过")


def main():
    print("=" * 60)
    print("  集成测试")
    print("=" * 60)

    test_state_manager()
    test_data_manager()
    test_agent_engine_init()

    print("\n✅ 全部集成测试通过！")


if __name__ == "__main__":
    main()
