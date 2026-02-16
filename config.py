import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env(key: str, default: str | None = None, required: bool = False) -> str:
    val = os.getenv(key, default)
    if required and not val:
        raise RuntimeError(f"Missing required env var: {key}")
    return val


def _env_float(key: str, default: float) -> float:
    return float(os.getenv(key, str(default)))


def _env_int(key: str, default: int) -> int:
    return int(os.getenv(key, str(default)))


def _env_bool(key: str, default: bool = False) -> bool:
    val = os.getenv(key, str(default)).lower()
    return val in ("true", "1", "yes")


@dataclass(frozen=True)
class AmberConfig:
    api_key: str = _env("AMBER_API_KEY", required=True)
    site_id: str = _env("AMBER_SITE_ID", "")  # auto-discovered if empty


@dataclass(frozen=True)
class SolcastConfig:
    api_key: str = _env("SOLCAST_API_KEY", required=True)
    resource_id: str = _env("SOLCAST_RESOURCE_ID", required=True)


@dataclass(frozen=True)
class FoxESSConfig:
    ip: str = _env("FOXESS_IP", required=True)
    port: int = _env_int("FOXESS_PORT", 502)
    slave_id: int = _env_int("FOXESS_SLAVE_ID", 247)


@dataclass(frozen=True)
class BatteryConfig:
    capacity_kwh: float = _env_float("BATTERY_CAPACITY_KWH", 42.0)
    max_power_kw: float = _env_float("BATTERY_MAX_POWER_KW", 10.0)
    min_soc_pct: float = _env_float("BATTERY_MIN_SOC_PCT", 10.0)
    efficiency: float = _env_float("BATTERY_EFFICIENCY", 0.90)
    cost_aud: float = _env_float("BATTERY_COST_AUD", 15000.0)
    cycle_life: int = _env_int("BATTERY_CYCLE_LIFE", 6000)

    @property
    def min_soc_kwh(self) -> float:
        return self.capacity_kwh * self.min_soc_pct / 100.0

    @property
    def cost_per_cycle(self) -> float:
        return self.cost_aud / self.cycle_life

    @property
    def degradation_per_kwh(self) -> float:
        """Cost in AUD per kWh of energy cycled through the battery."""
        return self.cost_per_cycle / self.capacity_kwh

    @property
    def one_way_efficiency(self) -> float:
        return self.efficiency ** 0.5


@dataclass(frozen=True)
class SystemConfig:
    scheduler_interval_s: int = _env_int("SCHEDULER_INTERVAL_SECONDS", 300)
    log_level: str = _env("LOG_LEVEL", "INFO")
    db_path: str = _env("DB_PATH", "power_arb.db")
    timezone: str = _env("TIMEZONE", "Australia/Brisbane")
    dry_run: bool = _env_bool("DRY_RUN", False)
    dashboard_enabled: bool = _env_bool("DASHBOARD_ENABLED", True)
    dashboard_port: int = _env_int("DASHBOARD_PORT", 8081)


# Singleton instances
amber = AmberConfig()
solcast = SolcastConfig()
foxess = FoxESSConfig()
battery = BatteryConfig()
system = SystemConfig()
