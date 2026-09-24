"""Tests for the WebSocket and REST sliding-window rate limiters."""

from types import SimpleNamespace

from api import rate_limit
from api.rate_limit import (
    MAX_MESSAGES,
    _rest_store,
    _timestamps,
    check_rate_limit,
    check_rest_rate_limit,
    clear_rate_limits,
    client_ip,
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


class TestClientIp:
    """The IP per-IP limits key on must come from the proxy, not the caller."""

    @staticmethod
    def _conn(forwarded=None, peer="10.0.0.1"):
        class _Conn:
            headers = {"x-forwarded-for": forwarded} if forwarded else {}
            client = SimpleNamespace(host=peer)
        return _Conn()

    def test_takes_rightmost_hop_not_leftmost(self):
        # The leftmost entry is client-supplied; the rightmost was appended by
        # our own edge. Picking the left one is the spoofing bug.
        assert client_ip(self._conn("1.1.1.1, 9.9.9.9")) == "9.9.9.9"

    def test_forged_prefix_does_not_change_the_key(self):
        a = client_ip(self._conn("11.11.11.11, 9.9.9.9"))
        b = client_ip(self._conn("22.22.22.22, 203.0.113.7, 9.9.9.9"))
        assert a == b == "9.9.9.9"

    def test_honours_extra_proxy_hops(self, monkeypatch):
        monkeypatch.setattr(rate_limit, "TRUSTED_PROXY_HOPS", 2)
        assert client_ip(self._conn("1.1.1.1, 203.0.113.7, 9.9.9.9")) == "203.0.113.7"

    def test_falls_back_to_peer_without_forwarded_header(self):
        assert client_ip(self._conn(peer="10.0.0.5")) == "10.0.0.5"

    def test_more_hops_configured_than_present(self, monkeypatch):
        # Never index past the chain into client-supplied text.
        monkeypatch.setattr(rate_limit, "TRUSTED_PROXY_HOPS", 5)
        assert client_ip(self._conn("1.1.1.1, 9.9.9.9")) == "1.1.1.1"

    def test_unparseable_hop_shares_one_bucket(self):
        # Fails closed: garbage must not mint a fresh quota per value.
        assert client_ip(self._conn("not-an-ip")) == "unknown"
        assert client_ip(self._conn("<script>")) == "unknown"

    def test_ipv6_hop(self):
        assert client_ip(self._conn("1.1.1.1, 2001:db8::1")) == "2001:db8::1"


class TestCheckRestRateLimit:
    def setup_method(self):
        clear_rate_limits()

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
        store = _rest_store(3600)
        store["filings:u:alice"] = [t - 3601 for t in store["filings:u:alice"]]
        assert check_rest_rate_limit("u:alice", "filings", 10, 3600) is True

    def test_cache_size_bounded_under_key_rotation(self):
        # 20k distinct keys → cache stays at or below its maxsize (10k).
        for i in range(20_000):
            check_rest_rate_limit(f"u:user_{i}", "filings", 10, 3600)
        assert len(_rest_store(3600)) <= 10_000

    def test_store_ttl_covers_its_window(self):
        # A store whose TTL is shorter than the window it serves would expire
        # entries early and silently hand the caller a fresh quota — the shape
        # of the 12h free-trial bug (entries evicted after 1h).
        for window in (3600, 43_200):
            assert _rest_store(window).ttl == window

    def test_long_window_quota_survives_short_window_traffic(self):
        # Long-window buckets get their own store, so short-window churn can't
        # evict them out of a shared maxsize.
        assert check_rest_rate_limit("ip:1.2.3.4", "anon_free_trial", 1, 43_200) is True
        for i in range(12_000):
            check_rest_rate_limit(f"u:user_{i}", "filings", 10, 3600)
        assert check_rest_rate_limit("ip:1.2.3.4", "anon_free_trial", 1, 43_200) is False

    def test_briefing_bucket_5_call_limit(self):
        # Mirrors the watchlist briefing endpoint config.
        for _ in range(5):
            assert check_rest_rate_limit("u:alice", "briefing", 5, 3600) is True
        assert check_rest_rate_limit("u:alice", "briefing", 5, 3600) is False
