"""Tests for Amber price forecast dampening."""

import os
import sys

os.environ.setdefault("AMBER_API_KEY", "psk_test")
os.environ.setdefault("SOLCAST_API_KEY", "test")
os.environ.setdefault("SOLCAST_RESOURCE_ID", "test")
os.environ.setdefault("FOXESS_IP", "127.0.0.1")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from datetime import datetime, timedelta
from amber.client import PriceInterval
from amber.price_dampener import PriceDampener
from storage.database import Database


def make_interval(
    per_kwh: float, minutes_ahead: int = 0,
    channel: str = "import", spike_status: str = "none"
) -> PriceInterval:
    now = datetime(2025, 6, 15, 14, 0, 0)
    ts = now + timedelta(minutes=minutes_ahead)
    return PriceInterval(
        timestamp=ts.isoformat(),
        end_time=(ts + timedelta(minutes=30)).isoformat(),
        per_kwh=per_kwh,
        spot_per_kwh=per_kwh * 0.5,
        channel=channel,
        forecast_type="forecast",
        spike_status=spike_status,
        duration_min=30,
    )


class TestPriceDampener:
    def setup_method(self):
        self.db = Database(":memory:")
        self.dampener = PriceDampener(self.db)
        self.ref_time = datetime(2025, 6, 15, 14, 0, 0)

    def test_short_term_passes_through(self):
        """Prices < 2h ahead should be mostly unchanged."""
        interval = make_interval(per_kwh=25.0, minutes_ahead=30)
        result = self.dampener.dampen([interval], self.ref_time)
        assert len(result) == 1
        assert abs(result[0].dampened_per_kwh - 25.0) < 1.0

    def test_negative_price_never_dampened(self):
        """Negative prices should pass through unchanged."""
        for ahead in [30, 180, 720, 1440]:
            interval = make_interval(per_kwh=-5.0, minutes_ahead=ahead)
            result = self.dampener.dampen([interval], self.ref_time)
            assert result[0].dampened_per_kwh == -5.0, \
                f"Negative price dampened at {ahead}min ahead"

    def test_extreme_forecast_dampened_medium_term(self):
        """Extreme prices 6h ahead should be significantly dampened."""
        interval = make_interval(per_kwh=2000.0, minutes_ahead=360)  # $20/kWh
        result = self.dampener.dampen([interval], self.ref_time)
        dampened = result[0].dampened_per_kwh
        # Should be much less than 2000, but more than typical (~10c)
        assert dampened < 500, f"Extreme price not dampened enough: {dampened}"
        assert dampened > 5, f"Extreme price over-dampened: {dampened}"

    def test_extreme_forecast_dampened_long_term(self):
        """Extreme prices 24h ahead should be heavily dampened toward historical."""
        interval = make_interval(per_kwh=2000.0, minutes_ahead=1440)
        result = self.dampener.dampen([interval], self.ref_time)
        dampened = result[0].dampened_per_kwh
        # 24h out: 0.3 * forecast + 0.7 * median => 0.3*2000 + 0.7*6 = ~604
        # Key check: it's far below the raw 2000c forecast
        assert dampened < 700, f"Long-term extreme not dampened: {dampened}"
        assert dampened > 5, f"Long-term extreme over-dampened: {dampened}"

    def test_spike_status_increases_dampening(self):
        """Spike status should cause more aggressive dampening."""
        normal = make_interval(per_kwh=500.0, minutes_ahead=180)
        spike = make_interval(per_kwh=500.0, minutes_ahead=180, spike_status="spike")
        potential = make_interval(per_kwh=500.0, minutes_ahead=180, spike_status="potential")

        r_normal = self.dampener.dampen([normal], self.ref_time)[0]
        r_spike = self.dampener.dampen([spike], self.ref_time)[0]
        r_potential = self.dampener.dampen([potential], self.ref_time)[0]

        # Potential should be dampened more than no-spike
        # Spike should be dampened more than potential (still confirmed but magnitude reduced)
        assert r_potential.dampened_per_kwh < r_normal.dampened_per_kwh
        assert r_spike.dampened_per_kwh < r_normal.dampened_per_kwh

    def test_confidence_decreases_with_lead_time(self):
        """Confidence should decrease as forecast gets further out."""
        intervals = [
            make_interval(per_kwh=25.0, minutes_ahead=m)
            for m in [30, 180, 720, 1440]
        ]
        results = self.dampener.dampen(intervals, self.ref_time)
        confidences = [r.confidence for r in results]
        # Each should be <= the previous (monotonically non-increasing)
        for i in range(1, len(confidences)):
            assert confidences[i] <= confidences[i - 1], \
                f"Confidence not decreasing: {confidences}"

    def test_moderate_price_less_dampened(self):
        """Normal-range prices should be dampened less than extreme ones."""
        moderate = make_interval(per_kwh=30.0, minutes_ahead=360)
        extreme = make_interval(per_kwh=2000.0, minutes_ahead=360)

        r_mod = self.dampener.dampen([moderate], self.ref_time)[0]
        r_ext = self.dampener.dampen([extreme], self.ref_time)[0]

        mod_ratio = r_mod.dampened_per_kwh / 30.0
        ext_ratio = r_ext.dampened_per_kwh / 2000.0
        # Extreme prices should be compressed proportionally more
        assert ext_ratio < mod_ratio

    def test_export_channel_dampened(self):
        """Export channel prices should also be dampened."""
        interval = make_interval(per_kwh=1000.0, minutes_ahead=360, channel="export")
        result = self.dampener.dampen([interval], self.ref_time)
        assert result[0].dampened_per_kwh < 1000.0
        assert result[0].channel == "export"
