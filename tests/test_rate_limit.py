"""Tests for per-IP WebSocket rate limiting."""

from api.rate_limit import check_rate_limit, _timestamps, MAX_MESSAGES


class TestRateLimit:
    def setup_method(self):
        _timestamps.clear()

    def test_allows_under_limit(self):
        for _ in range(MAX_MESSAGES):
            assert check_rate_limit("1.2.3.4") is True

    def test_blocks_over_limit(self):
        for _ in range(MAX_MESSAGES):
            check_rate_limit("1.2.3.4")

        assert check_rate_limit("1.2.3.4") is False

    def test_separate_ips(self):
        for _ in range(MAX_MESSAGES):
            check_rate_limit("1.1.1.1")

        # Different IP should still be allowed
        assert check_rate_limit("2.2.2.2") is True

    def test_window_expiry(self):
        import time
        from unittest.mock import patch

        # Fill up the limit
        for _ in range(MAX_MESSAGES):
            check_rate_limit("1.2.3.4")
        assert check_rate_limit("1.2.3.4") is False

        # Fast-forward past the window by manipulating timestamps
        _timestamps["1.2.3.4"] = [t - 61 for t in _timestamps["1.2.3.4"]]
        assert check_rate_limit("1.2.3.4") is True
