import logging
from dataclasses import dataclass
from datetime import datetime, date, timezone

from amberelectric import AmberApi, ApiClient, Configuration, ChannelType, SpikeStatus

import config
from storage.database import Database

logger = logging.getLogger(__name__)

@dataclass
class PriceInterval:
    timestamp: str          # ISO 8601 start time
    end_time: str           # ISO 8601 end time
    per_kwh: float          # cents/kWh total (what you pay/receive)
    spot_per_kwh: float     # cents/kWh wholesale component
    channel: str            # 'import' or 'export'
    forecast_type: str      # 'actual', 'current', 'forecast'
    spike_status: str       # 'none', 'potential', 'spike'
    duration_min: int       # interval duration in minutes


class AmberClient:
    def __init__(self, db: Database):
        self.db = db
        configuration = Configuration(access_token=config.amber.api_key)
        api_client = ApiClient(configuration=configuration)
        self.api = AmberApi(api_client=api_client)
        self.site_id = config.amber.site_id or self._discover_site_id()

    def _discover_site_id(self) -> str:
        sites = self.api.get_sites()
        if not sites:
            raise RuntimeError("No Amber sites found for this API key")
        site_id = sites[0].id
        logger.info("Auto-discovered Amber site: %s", site_id)
        return site_id

    def fetch_current_and_forecast(self) -> dict[str, list[PriceInterval]]:
        """Fetch current + 48h forecast prices at 5-min resolution.

        Returns dict with keys 'import' and 'export', each containing a list
        of PriceInterval sorted by timestamp.
        """
        raw = self.api.get_current_prices(self.site_id, next=576, resolution=5)
        now = datetime.now().isoformat()

        import_prices = []
        export_prices = []
        db_records = []


        for interval in raw:

            #print(type(interval))
            #print(vars(interval))
            
            channel_type = interval.actual_instance.channel_type
            if channel_type == ChannelType.GENERAL:
                channel = "import"
            elif channel_type == ChannelType.FEEDIN:
                channel = "export"
            else:
                continue  # skip controlled load

            interval_type = str(interval.actual_instance.type).lower()
            if "current" in interval_type:
                forecast_type = "current"
            elif "forecast" in interval_type:
                forecast_type = "forecast"
            else:
                forecast_type = "actual"

            # spike_status is a SpikeStatus enum; extract its string value
            raw_spike = getattr(interval, "spike_status", None)
            spike_str = raw_spike.value if raw_spike else "none"

            pi = PriceInterval(
                timestamp=interval.actual_instance.start_time.isoformat(),
                end_time=interval.actual_instance.end_time.isoformat(),
                per_kwh=interval.actual_instance.per_kwh,
                spot_per_kwh=getattr(interval.actual_instance, "spot_per_kwh", None) or 0.0,
                channel=channel,
                forecast_type=forecast_type,
                spike_status=spike_str,
                duration_min=interval.actual_instance.duration or 5,
            )

            if channel == "import":
                import_prices.append(pi)
            else:
                export_prices.append(pi)

            db_records.append({
                "timestamp": pi.timestamp,
                "channel": channel,
                "per_kwh": pi.per_kwh,
                "spot_per_kwh": pi.spot_per_kwh,
                "forecast_type": forecast_type,
                "spike_status": pi.spike_status,
                "fetched_at": now,
            })

        # Store raw prices
        if db_records:
            self.db.insert_prices(db_records)
            logger.info(
                "Fetched %d import + %d export price intervals",
                len(import_prices), len(export_prices),
            )

        # Store forecast accuracy records for later calibration
        self._record_forecast_accuracy(import_prices, "import", now)
        self._record_forecast_accuracy(export_prices, "export", now)

        # Backfill actuals into past forecast records
        self._backfill_actuals(import_prices, "import")
        self._backfill_actuals(export_prices, "export")

        return {
            "import": sorted(import_prices, key=lambda p: p.timestamp),
            "export": sorted(export_prices, key=lambda p: p.timestamp),
        }

    def _record_forecast_accuracy(
        self, prices: list[PriceInterval], channel: str, fetch_time: str
    ):
        """Store forecast prices for later comparison with actuals."""
        records = []
        now = datetime.fromisoformat(fetch_time)
        if now.tzinfo is None: now = now.replace(tzinfo=timezone.utc)
        else: now = now.astimezone(timezone.utc)
        
        for p in prices:
            if p.forecast_type != "forecast":
                continue
            target = datetime.fromisoformat(p.timestamp)
            lead_minutes = int((target - now).total_seconds() / 60)
            if lead_minutes < 0:
                continue
            records.append({
                "forecast_time": fetch_time,
                "target_time": p.timestamp,
                "forecast_price": p.per_kwh,
                "actual_price": None,
                "channel": channel,
                "lead_time_minutes": lead_minutes,
            })
        if records:
            self.db.insert_forecast_accuracy(records)

    def _backfill_actuals(self, prices: list[PriceInterval], channel: str):
        """Update forecast accuracy records with actual prices as they arrive."""
        for p in prices:
            if p.forecast_type in ("actual", "current"):
                self.db.update_forecast_actuals(p.timestamp, channel, p.per_kwh)

    def fetch_historical(self, start: date, end: date) -> list[dict]:
        """Fetch historical prices for backtesting or analysis."""
        raw = self.api.get_prices(self.site_id, start_date=start, end_date=end)
        results = []
        for interval in raw:
            channel_type = interval.channel_type
            if channel_type == ChannelType.GENERAL:
                channel = "import"
            elif channel_type == ChannelType.FEEDIN:
                channel = "export"
            else:
                continue
            results.append({
                "timestamp": interval.start_time.isoformat(),
                "channel": channel,
                "per_kwh": interval.per_kwh,
                "spot_per_kwh": getattr(interval, "spot_per_kwh", None) or 0.0,
            })
        return results
