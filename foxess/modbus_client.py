import logging
import time
from dataclasses import dataclass
from enum import IntEnum

from pymodbus.client import ModbusTcpClient
from pymodbus.exceptions import ModbusException

import config

logger = logging.getLogger(__name__)

# Minimum delay between Modbus write operations (seconds)
_WRITE_DELAY = 0.25
# Delay before read-back verification after a write
_VERIFY_DELAY = 0.5


class WorkMode(IntEnum):
    SELF_USE = 0
    FEED_IN_FIRST = 1
    BACKUP = 2
    FORCE_CHARGE = 3
    FORCE_DISCHARGE = 4


# -- Register addresses (0-based, FoxESS H1/H3/KH series) --
# IMPORTANT: Verify these against your specific model/firmware.
# Reference: https://github.com/nathanmarlor/foxess_modbus

class InputRegisters:
    """Read-only input registers (function code 0x04)."""
    PV1_POWER = 11002
    PV2_POWER = 11005
    GRID_CT_POWER = 11021
    LOAD_POWER = 11023       # H1 series; H3 uses 11025
    BATTERY_VOLTAGE = 11033
    BATTERY_POWER = 11034
    BATTERY_CURRENT = 11035
    BATTERY_SOC = 11036
    BATTERY_TEMP = 11038


class HoldingRegisters:
    """Read/write holding registers (function code 0x03/0x06)."""
    WORK_MODE = 41000
    TIME1_ENABLE = 41001
    TIME1_START_HOUR = 41002
    TIME1_START_MIN = 41003
    TIME1_END_HOUR = 41004
    TIME1_END_MIN = 41005
    TIME1_CHARGE_POWER = 41006
    TIME1_DISCHARGE_POWER = 41007
    MIN_SOC = 41009
    MIN_SOC_ON_GRID = 41010
    EXPORT_LIMIT = 41012


@dataclass
class InverterState:
    """Snapshot of current inverter/battery/grid state."""
    soc_pct: float        # battery state of charge (0-100%)
    soc_kwh: float        # derived from soc_pct and battery capacity
    battery_power_w: int  # positive = charging, negative = discharging
    pv_power_w: int       # total solar generation
    grid_power_w: int     # positive = importing, negative = exporting
    load_power_w: int     # home consumption
    battery_temp_c: float
    work_mode: WorkMode
    min_soc_pct: int
    export_limit_w: int


def _to_signed16(value: int) -> int:
    """Convert unsigned uint16 from Modbus to signed int16."""
    return value - 65536 if value > 32767 else value


class FoxESSModbusClient:
    """Controls FoxESS inverter via local Modbus TCP.

    Safety rules:
    - 250ms minimum between writes
    - Read-back verification after every write
    - Never set min SoC below 10%
    - Reverts to self-use mode on communication failure
    """

    def __init__(self):
        self.ip = config.foxess.ip
        self.port = config.foxess.port
        self.slave_id = config.foxess.slave_id
        self.dry_run = config.system.dry_run
        self._client: ModbusTcpClient | None = None
        self._last_write_time = 0.0

    def connect(self) -> bool:
        if self._client and self._client.connected:
            return True
        self._client = ModbusTcpClient(
            host=self.ip, port=self.port, timeout=5, retries=3
        )
        connected = self._client.connect()
        if connected:
            logger.info("Connected to FoxESS inverter at %s:%s", self.ip, self.port)
        else:
            logger.error("Failed to connect to FoxESS inverter at %s:%s", self.ip, self.port)
        return connected

    def disconnect(self):
        if self._client:
            self._client.close()
            self._client = None

    def read_state(self) -> InverterState | None:
        """Read current inverter state from input and holding registers.

        Reads are always performed, even in dry-run mode, so the optimizer
        works with real SoC, PV, load and grid data.
        """
        if not self.connect():
            return None

        try:
            # Read input registers (battery, PV, grid, load)
            ir = self._read_input_registers(InputRegisters.PV1_POWER, 2)
            pv1_power = _to_signed16(ir[0]) if ir else 0
            pv2_result = self._read_input_registers(InputRegisters.PV2_POWER, 1)
            pv2_power = _to_signed16(pv2_result[0]) if pv2_result else 0

            grid_result = self._read_input_registers(InputRegisters.GRID_CT_POWER, 1)
            grid_power = _to_signed16(grid_result[0]) if grid_result else 0

            load_result = self._read_input_registers(InputRegisters.LOAD_POWER, 1)
            load_power = _to_signed16(load_result[0]) if load_result else 0

            bat_result = self._read_input_registers(InputRegisters.BATTERY_POWER, 4)
            if bat_result:
                battery_power = _to_signed16(bat_result[0])
                battery_soc = bat_result[2]  # offset 2 = register 11036
                battery_temp = _to_signed16(bat_result[4]) / 10.0 if len(bat_result) > 4 else 0.0
            else:
                battery_power, battery_soc, battery_temp = 0, 0, 0.0

            # Read holding registers (work mode, min SoC)
            hr = self._read_holding_registers(HoldingRegisters.WORK_MODE, 1)
            work_mode = WorkMode(hr[0]) if hr else WorkMode.SELF_USE

            min_soc_result = self._read_holding_registers(HoldingRegisters.MIN_SOC, 1)
            min_soc = min_soc_result[0] if min_soc_result else 10

            export_limit_result = self._read_holding_registers(HoldingRegisters.EXPORT_LIMIT, 1)
            export_limit_w = export_limit_result[0] if export_limit_result else 10000

            soc_kwh = battery_soc / 100.0 * config.battery.capacity_kwh

            return InverterState(
                soc_pct=battery_soc,
                soc_kwh=soc_kwh,
                battery_power_w=battery_power,
                pv_power_w=pv1_power + pv2_power,
                grid_power_w=grid_power,
                load_power_w=load_power,
                battery_temp_c=battery_temp,
                work_mode=work_mode,
                min_soc_pct=min_soc,
                export_limit_w=export_limit_w,
            )

        except ModbusException as e:
            logger.error("Modbus read error: %s", e)
            return None

    def set_work_mode(self, mode: WorkMode) -> bool:
        """Set the inverter work mode with verification."""
        logger.info("Setting work mode to %s (%d)", mode.name, mode.value)
        if not self._write_register(HoldingRegisters.WORK_MODE, mode.value):
            return False

        if self.dry_run:
            return True

        # Verify
        time.sleep(_VERIFY_DELAY)
        result = self._read_holding_registers(HoldingRegisters.WORK_MODE, 1)
        if result and result[0] == mode.value:
            logger.info("Work mode verified: %s", mode.name)
            return True
        else:
            logger.error("Work mode verification failed! Expected %d, got %s",
                         mode.value, result)
            return False

    def set_min_soc(self, soc_pct: int) -> bool:
        """Set minimum SoC percentage (clamped to 10-100%)."""
        soc_pct = max(10, min(100, soc_pct))
        logger.info("Setting min SoC to %d%%", soc_pct)
        if not self._write_register(HoldingRegisters.MIN_SOC, soc_pct):
            return False
        # Also set min SoC on grid to same value
        self._write_register(HoldingRegisters.MIN_SOC_ON_GRID, soc_pct)
        return True

    def set_export_limit(self, power_w: int) -> bool:
        """Set grid export power limit in watts (clamped to 0-10000W)."""
        power_w = max(0, min(10000, power_w))
        logger.info("Setting export limit to %dW", power_w)
        return self._write_register(HoldingRegisters.EXPORT_LIMIT, power_w)

    def set_charge_period(
        self, enable: bool, start_hour: int, start_min: int,
        end_hour: int, end_min: int,
        charge_power_w: int = 10000, discharge_power_w: int = 0
    ) -> bool:
        """Configure time period 1 for scheduled charge/discharge."""
        writes = [
            (HoldingRegisters.TIME1_ENABLE, 1 if enable else 0),
            (HoldingRegisters.TIME1_START_HOUR, start_hour),
            (HoldingRegisters.TIME1_START_MIN, start_min),
            (HoldingRegisters.TIME1_END_HOUR, end_hour),
            (HoldingRegisters.TIME1_END_MIN, end_min),
            (HoldingRegisters.TIME1_CHARGE_POWER, charge_power_w),
            (HoldingRegisters.TIME1_DISCHARGE_POWER, discharge_power_w),
        ]
        for addr, val in writes:
            if not self._write_register(addr, val):
                return False
        return True

    def emergency_self_use(self):
        """Force inverter back to self-use mode. Called on errors or watchdog timeout."""
        logger.warning("EMERGENCY: Forcing self-use mode")
        try:
            if self.connect():
                self._write_register(HoldingRegisters.WORK_MODE, WorkMode.SELF_USE.value)
                self._write_register(HoldingRegisters.MIN_SOC, 10)
        except Exception as e:
            logger.error("Emergency self-use failed: %s", e)

    # -- Low-level Modbus operations --

    def _read_input_registers(self, address: int, count: int) -> list[int] | None:
        try:
            result = self._client.read_input_registers(
                address=address, count=count, slave=self.slave_id
            )
            if result.isError():
                logger.error("Read input register %d error: %s", address, result)
                return None
            return result.registers
        except Exception as e:
            logger.error("Read input register %d exception: %s", address, e)
            return None

    def _read_holding_registers(self, address: int, count: int) -> list[int] | None:
        try:
            result = self._client.read_holding_registers(
                address=address, count=count, slave=self.slave_id
            )
            if result.isError():
                logger.error("Read holding register %d error: %s", address, result)
                return None
            return result.registers
        except Exception as e:
            logger.error("Read holding register %d exception: %s", address, e)
            return None

    def _write_register(self, address: int, value: int) -> bool:
        if self.dry_run:
            logger.info("[DRY RUN] Would write register %d = %d", address, value)
            return True

        if not self.connect():
            return False

        # Enforce minimum delay between writes
        elapsed = time.time() - self._last_write_time
        if elapsed < _WRITE_DELAY:
            time.sleep(_WRITE_DELAY - elapsed)

        try:
            result = self._client.write_register(
                address=address, value=value, slave=self.slave_id
            )
            self._last_write_time = time.time()
            if result.isError():
                logger.error("Write register %d = %d error: %s", address, value, result)
                return False
            return True
        except Exception as e:
            logger.error("Write register %d = %d exception: %s", address, value, e)
            return False
