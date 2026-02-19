import sqlite3
import logging
from contextlib import contextmanager
from datetime import datetime, date
from pathlib import Path

import config

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    channel TEXT NOT NULL,          -- 'import' or 'export'
    per_kwh REAL NOT NULL,         -- cents/kWh AUD (total price)
    spot_per_kwh REAL,             -- cents/kWh AUD (wholesale component)
    forecast_type TEXT NOT NULL,   -- 'actual', 'current', 'forecast'
    spike_status TEXT,             -- 'none', 'potential', 'spike'
    fetched_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_prices_ts ON prices(timestamp, channel);
CREATE INDEX IF NOT EXISTS idx_prices_fetched ON prices(fetched_at);

CREATE TABLE IF NOT EXISTS solar_forecasts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fetch_time TEXT NOT NULL,
    period_end TEXT NOT NULL,
    pv_estimate_kw REAL NOT NULL,
    pv_estimate10_kw REAL,
    pv_estimate90_kw REAL
);

CREATE INDEX IF NOT EXISTS idx_solar_period ON solar_forecasts(period_end);
CREATE INDEX IF NOT EXISTS idx_solar_fetch ON solar_forecasts(fetch_time);

CREATE TABLE IF NOT EXISTS consumption_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    load_watts REAL NOT NULL,
    source TEXT NOT NULL DEFAULT 'measured'  -- 'measured' or 'predicted'
);

CREATE INDEX IF NOT EXISTS idx_consumption_ts ON consumption_log(timestamp);

CREATE TABLE IF NOT EXISTS battery_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    soc_pct REAL NOT NULL,
    soc_kwh REAL NOT NULL,
    battery_power_w REAL NOT NULL,
    grid_power_w REAL NOT NULL,
    pv_power_w REAL NOT NULL,
    load_power_w REAL NOT NULL,
    battery_temp_c REAL
);

CREATE INDEX IF NOT EXISTS idx_battery_ts ON battery_log(timestamp);

CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    action TEXT NOT NULL,
    reason TEXT,
    import_price REAL,
    export_price REAL,
    dampened_import_price REAL,
    dampened_export_price REAL,
    soc_kwh REAL,
    expected_profit REAL,
    actual_mode_set INTEGER
);

CREATE INDEX IF NOT EXISTS idx_decisions_ts ON decisions(timestamp);

CREATE TABLE IF NOT EXISTS forecast_accuracy (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    forecast_time TEXT NOT NULL,
    target_time TEXT NOT NULL,
    forecast_price REAL NOT NULL,
    actual_price REAL,
    channel TEXT NOT NULL,
    lead_time_minutes INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_accuracy_target ON forecast_accuracy(target_time, channel);

CREATE TABLE IF NOT EXISTS weather_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fetch_time TEXT NOT NULL,
    target_time TEXT NOT NULL,
    temperature_c REAL NOT NULL,
    sunrise TEXT,
    sunset TEXT
);

CREATE INDEX IF NOT EXISTS idx_weather_target ON weather_cache(target_time);
CREATE INDEX IF NOT EXISTS idx_weather_fetch ON weather_cache(fetch_time);

CREATE TABLE IF NOT EXISTS feature_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    snapshot_type TEXT NOT NULL,  -- 'price' or 'consumption'
    features_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_features_ts ON feature_snapshots(timestamp, snapshot_type);

CREATE TABLE IF NOT EXISTS occupancy_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    anyone_home INTEGER NOT NULL,
    entities_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_occupancy_ts ON occupancy_log(timestamp);
"""


class Database:
    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or config.system.db_path
        self._persistent_conn: sqlite3.Connection | None = None
        # For :memory: databases, keep a single connection alive
        if self.db_path == ":memory:":
            self._persistent_conn = sqlite3.connect(":memory:")
            self._persistent_conn.row_factory = sqlite3.Row
            self._persistent_conn.execute("PRAGMA foreign_keys=ON")
        self._init_db()

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            # Migrations for existing databases
            self._migrate(conn)
            logger.info("Database initialized at %s", self.db_path)

    def _migrate(self, conn):
        """Add columns that may be missing from older schemas."""
        cols = {r[1] for r in conn.execute("PRAGMA table_info(battery_log)").fetchall()}
        if "battery_temp_c" not in cols:
            conn.execute("ALTER TABLE battery_log ADD COLUMN battery_temp_c REAL")
            logger.info("Migrated battery_log: added battery_temp_c column")

    @contextmanager
    def _connect(self):
        if self._persistent_conn:
            # In-memory DB: reuse the persistent connection
            try:
                yield self._persistent_conn
                self._persistent_conn.commit()
            except Exception:
                self._persistent_conn.rollback()
                raise
        else:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    # -- Price operations --

    def insert_prices(self, prices: list[dict]):
        with self._connect() as conn:
            conn.executemany(
                """INSERT INTO prices
                   (timestamp, channel, per_kwh, spot_per_kwh,
                    forecast_type, spike_status, fetched_at)
                   VALUES (:timestamp, :channel, :per_kwh, :spot_per_kwh,
                           :forecast_type, :spike_status, :fetched_at)""",
                prices,
            )

    def get_prices(
        self, start: str, end: str, channel: str = "import"
    ) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM prices
                   WHERE timestamp >= ? AND timestamp <= ? AND channel = ?
                   ORDER BY timestamp""",
                (start, end, channel),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_latest_prices(self, channel: str = "import") -> list[dict]:
        """Get the most recently fetched set of prices (current + forecast)."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT MAX(fetched_at) as latest FROM prices WHERE channel = ?",
                (channel,),
            ).fetchone()
            if not row or not row["latest"]:
                return []
            rows = conn.execute(
                """SELECT * FROM prices
                   WHERE fetched_at = ? AND channel = ?
                   ORDER BY timestamp""",
                (row["latest"], channel),
            ).fetchall()
            return [dict(r) for r in rows]

    # -- Solar forecast operations --

    def insert_solar_forecasts(self, forecasts: list[dict]):
        with self._connect() as conn:
            conn.executemany(
                """INSERT INTO solar_forecasts
                   (fetch_time, period_end, pv_estimate_kw,
                    pv_estimate10_kw, pv_estimate90_kw)
                   VALUES (:fetch_time, :period_end, :pv_estimate_kw,
                           :pv_estimate10_kw, :pv_estimate90_kw)""",
                forecasts,
            )

    def get_latest_solar_forecast(self) -> list[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT MAX(fetch_time) as latest FROM solar_forecasts"
            ).fetchone()
            if not row or not row["latest"]:
                return []
            rows = conn.execute(
                """SELECT * FROM solar_forecasts
                   WHERE fetch_time = ?
                   ORDER BY period_end""",
                (row["latest"],),
            ).fetchall()
            return [dict(r) for r in rows]

    # -- Consumption operations --

    def insert_consumption(self, timestamp: str, load_watts: float, source: str = "measured"):
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO consumption_log (timestamp, load_watts, source)
                   VALUES (?, ?, ?)""",
                (timestamp, load_watts, source),
            )

    def get_consumption_history(self, days: int = 14) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM consumption_log
                   WHERE source = 'measured'
                   AND timestamp >= datetime('now', ?)
                   ORDER BY timestamp""",
                (f"-{days} days",),
            ).fetchall()
            return [dict(r) for r in rows]

    # -- Battery log operations --

    def insert_battery_log(self, entry: dict):
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO battery_log
                   (timestamp, soc_pct, soc_kwh, battery_power_w,
                    grid_power_w, pv_power_w, load_power_w, battery_temp_c)
                   VALUES (:timestamp, :soc_pct, :soc_kwh, :battery_power_w,
                           :grid_power_w, :pv_power_w, :load_power_w, :battery_temp_c)""",
                entry,
            )

    def get_battery_log_since(self, since: str) -> list[dict]:
        """Get battery log entries since a given timestamp."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM battery_log WHERE timestamp >= ? ORDER BY timestamp",
                (since,),
            ).fetchall()
            return [dict(r) for r in rows]

    # -- Decision operations --

    def insert_decision(self, decision: dict):
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO decisions
                   (timestamp, action, reason, import_price, export_price,
                    dampened_import_price, dampened_export_price,
                    soc_kwh, expected_profit, actual_mode_set)
                   VALUES (:timestamp, :action, :reason, :import_price,
                           :export_price, :dampened_import_price,
                           :dampened_export_price, :soc_kwh,
                           :expected_profit, :actual_mode_set)""",
                decision,
            )

    def get_decisions_since(self, since: str) -> list[dict]:
        """Get decision records since a given timestamp."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM decisions WHERE timestamp >= ? ORDER BY timestamp",
                (since,),
            ).fetchall()
            return [dict(r) for r in rows]

    # -- Forecast accuracy operations --

    def insert_forecast_accuracy(self, records: list[dict]):
        with self._connect() as conn:
            conn.executemany(
                """INSERT INTO forecast_accuracy
                   (forecast_time, target_time, forecast_price,
                    actual_price, channel, lead_time_minutes)
                   VALUES (:forecast_time, :target_time, :forecast_price,
                           :actual_price, :channel, :lead_time_minutes)""",
                records,
            )

    def get_forecast_accuracy(self, channel: str = "import", days: int = 30) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM forecast_accuracy
                   WHERE channel = ? AND actual_price IS NOT NULL
                   AND target_time >= datetime('now', ?)
                   ORDER BY target_time""",
                (channel, f"-{days} days"),
            ).fetchall()
            return [dict(r) for r in rows]

    def update_forecast_actuals(self, target_time: str, channel: str, actual_price: float):
        """Backfill actual prices into forecast accuracy records."""
        with self._connect() as conn:
            conn.execute(
                """UPDATE forecast_accuracy
                   SET actual_price = ?
                   WHERE target_time = ? AND channel = ? AND actual_price IS NULL""",
                (actual_price, target_time, channel),
            )

    # -- Weather cache operations --

    def insert_weather_cache(self, records: list[dict]):
        with self._connect() as conn:
            conn.executemany(
                """INSERT INTO weather_cache
                   (fetch_time, target_time, temperature_c, sunrise, sunset)
                   VALUES (:fetch_time, :target_time, :temperature_c, :sunrise, :sunset)""",
                records,
            )

    def get_latest_weather(self) -> list[dict]:
        """Get the most recently fetched weather forecast."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT MAX(fetch_time) as latest FROM weather_cache"
            ).fetchone()
            if not row or not row["latest"]:
                return []
            rows = conn.execute(
                "SELECT * FROM weather_cache WHERE fetch_time = ? ORDER BY target_time",
                (row["latest"],),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_weather_at(self, target_time: str) -> dict | None:
        """Get weather closest to a target time from the latest fetch."""
        with self._connect() as conn:
            row = conn.execute(
                """SELECT * FROM weather_cache
                   WHERE fetch_time = (SELECT MAX(fetch_time) FROM weather_cache)
                   ORDER BY ABS(julianday(target_time) - julianday(?))
                   LIMIT 1""",
                (target_time,),
            ).fetchone()
            return dict(row) if row else None

    # -- Feature snapshot operations --

    def insert_feature_snapshot(self, timestamp: str, snapshot_type: str, features: dict):
        import json
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO feature_snapshots (timestamp, snapshot_type, features_json)
                   VALUES (?, ?, ?)""",
                (timestamp, snapshot_type, json.dumps(features)),
            )

    def get_feature_snapshots(self, snapshot_type: str, days: int = 60) -> list[dict]:
        import json
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM feature_snapshots
                   WHERE snapshot_type = ? AND timestamp >= datetime('now', ?)
                   ORDER BY timestamp""",
                (snapshot_type, f"-{days} days"),
            ).fetchall()
            result = []
            for r in rows:
                d = dict(r)
                d["features"] = json.loads(d["features_json"])
                result.append(d)
            return result

    # -- Occupancy operations --

    def insert_occupancy(self, timestamp: str, anyone_home: bool, entities: dict | None = None):
        import json
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO occupancy_log (timestamp, anyone_home, entities_json)
                   VALUES (?, ?, ?)""",
                (timestamp, int(anyone_home), json.dumps(entities) if entities else None),
            )

    def get_latest_occupancy(self) -> bool:
        """Return most recent occupancy state. Defaults to True if no data."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT anyone_home FROM occupancy_log ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            return bool(row["anyone_home"]) if row else True

    def get_occupancy_history(self, hours: int = 24) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM occupancy_log
                   WHERE timestamp >= datetime('now', ?)
                   ORDER BY timestamp""",
                (f"-{hours} hours",),
            ).fetchall()
            return [dict(r) for r in rows]

    # -- Maintenance --

    def prune_old_records(self, days: int = 365):
        """Delete records older than `days` to prevent unbounded DB growth.

        Preserves the decisions table (useful for long-term analysis).
        """
        cutoff = f"-{days} days"
        with self._connect() as conn:
            for table, col in [
                ("battery_log", "timestamp"),
                ("consumption_log", "timestamp"),
                ("forecast_accuracy", "target_time"),
                ("prices", "fetched_at"),
                ("solar_forecasts", "fetch_time"),
                ("weather_cache", "fetch_time"),
                ("feature_snapshots", "timestamp"),
                ("occupancy_log", "timestamp"),
            ]:
                result = conn.execute(
                    f"DELETE FROM {table} WHERE {col} < datetime('now', ?)",
                    (cutoff,),
                )
                if result.rowcount > 0:
                    logger.info("Pruned %d rows from %s (older than %d days)",
                                result.rowcount, table, days)
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
