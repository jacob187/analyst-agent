"""Tests for the WebSocket and REST sliding-window rate limiters."""

from api.rate_limit import (
    MAX_MESSAGES,
    _rest_timestamps,
    _timestamps,
    check_rate_limit,
    check_rest_rate_limit,
    rate_limit_key,
)


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
        # Fill up the limit
        for _ in range(MAX_MESSAGES):
            check_rate_limit("1.2.3.4")
        assert check_rate_limit("1.2.3.4") is False

        # Fast-forward past the window by manipulating stored timestamps
        _timestamps["1.2.3.4"] = [t - 61 for t in _timestamps["1.2.3.4"]]
        assert check_rate_limit("1.2.3.4") is True


class TestRateLimitKey:
    def test_user_id_preferred_over_ip(self):
        assert rate_limit_key("user_abc", "1.2.3.4") == "u:user_abc"

    def test_falls_back_to_ip_when_no_user_id(self):
        assert rate_limit_key(None, "1.2.3.4") == "ip:1.2.3.4"

    def test_user_and_ip_keys_dont_collide(self):
        assert rate_limit_key("1.2.3.4", "1.2.3.4") != rate_limit_key(None, "1.2.3.4")


class TestCheckRestRateLimit:
    def setup_method(self):
        _rest_timestamps.clear()

    def test_allows_under_limit(self):
        for _ in range(9):
            assert check_rest_rate_limit("u:alice", "filings", 10, 3600) is True

    def test_blocks_at_limit(self):
        for _ in range(10):
            check_rest_rate_limit("u:alice", "filings", 10, 3600)
        assert check_rest_rate_limit("u:alice", "filings", 10, 3600) is False

    def test_independent_buckets(self):
        for _ in range(10):
            check_rest_rate_limit("u:alice", "filings", 10, 3600)
        # Same key, different bucket — quota is separate
        assert check_rest_rate_limit("u:alice", "briefing", 5, 3600) is True

    def test_independent_keys(self):
        for _ in range(10):
            check_rest_rate_limit("u:alice", "filings", 10, 3600)
        assert check_rest_rate_limit("u:bob", "filings", 10, 3600) is True

    def test_window_expiry(self):
        for _ in range(10):
            check_rest_rate_limit("u:alice", "filings", 10, 3600)
        assert check_rest_rate_limit("u:alice", "filings", 10, 3600) is False

        # Mutate timestamps to look old; next call should be allowed again.
        _rest_timestamps["filings:u:alice"] = [
            t - 3601 for t in _rest_timestamps["filings:u:alice"]
        ]
        assert check_rest_rate_limit("u:alice", "filings", 10, 3600) is True

    def test_cache_size_bounded_under_key_rotation(self):
        # 20k distinct keys → cache stays at or below its maxsize (10k).
        for i in range(20_000):
            check_rest_rate_limit(f"u:user_{i}", "filings", 10, 3600)
        assert len(_rest_timestamps) <= 10_000

    def test_briefing_bucket_5_call_limit(self):
        # Mirrors the watchlist briefing endpoint config.
        for _ in range(5):
            assert check_rest_rate_limit("u:alice", "briefing", 5, 3600) is True
        assert check_rest_rate_limit("u:alice", "briefing", 5, 3600) is False
