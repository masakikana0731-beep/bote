from boatbot.model import Settings
from boatbot.store import Store


def test_budget_counting(tmp_path):
    st = Store(tmp_path / "t.db")
    a = st.add_rec("20261013", "01", 1, "15:00", "", "pending", [{"combo": "1-2-3", "amount": 300}], 300, {})
    b = st.add_rec("20261013", "01", 2, "15:30", "", "pending", [{"combo": "1-2-3", "amount": 500}], 500, {})
    st.add_rec("20261013", "01", 3, "16:00", "", "pass", [], 0, {})
    assert st.spent("20261013") == (800, 2)
    st.set_status(b, "skipped")
    assert st.spent("20261013") == (300, 1)
    st.set_status(a, "bought")
    st.set_result(a, "1-2-3", 1000, 3000)
    day = st.stats()[0]
    assert day["real_bet"] == 300 and day["real_ret"] == 3000 and day["hits"] == 1


def test_settings_roundtrip(tmp_path):
    st = Store(tmp_path / "t.db")
    st.save_settings(Settings.from_dict({"gamble": 150, "daily_budget": 5000, "venues": "1, 12"}))
    s = st.settings()
    assert s.gamble == 100 and s.daily_budget == 5000 and s.venue_set() == {"01", "12"}
