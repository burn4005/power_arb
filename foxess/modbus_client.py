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
    """KH series only supports three work modes (0-2).
    Modes 3 (FORCE_CHARGE) and 4 (FORCE_DISCHARGE) do not exist on KH —
    use remote control registers 44000-44002 instead.
    """
    SELF_USE = 0
    FEED_IN_FIRST = 1
    BACKUP = 2


# ---------------------------------------------------------------------------
# KH Series Register Map (all FC 0x03 holding registers)
# Reference: https://github.com/nathanmarlor/foxess_modbus  (KH_PRE133 profile)
# ---------------------------------------------------------------------------

class KHRegisters:
    """KH series holding registers (FC 0x03 read, FC 0x06 write).

    Monitoring registers (read-only in practice):
      31002–31005  PV power
      31020–31029  Battery parameters + BMS
      31049–31054  Grid CT (32-bit) + load

    Control registers (read/write):
      41000        Work mode (0=Self-Use, 1=Feed-In First, 2=Back-up)
      41009        Min SoC %
      41010        Min SoC on grid %
      44000        Remote control enable (0=off, 1=on)
      44001        Remote control watchdog timeout (ms)
      44002        Remote control active power (W, negative=charge, positive=discharge)
    """
    # -- PV (batch B: 31002–31005, 4 registers) --
    PV1_VOLTAGE   = 31000   # ÷10 = V
    PV1_CURRENT   = 31001   # ÷10 = A
    PV1_POWER     = 31002   # raw = W
    PV2_VOLTAGE   = 31003   # ÷10 = V
    PV2_CURRENT   = 31004   # ÷10 = A
    PV2_POWER     = 31005   # raw = W

    # -- Battery / BMS (batch A: 31020–31029, 10 registers) --
    BATTERY_VOLTAGE      = 31020  # ÷10 = V
    BATTERY_CURRENT      = 31021  # ÷10 = A  (positive=charging, negative=discharging)
    BATTERY_POWER        = 31022  # raw = W  (positive=charging, negative=discharging)
    BATTERY_TEMP         = 31023  # ÷10 = °C
    BATTERY_SOC          = 31024  # % direct (0-100)
    BMS_MAX_CHARGE_A     = 31025  # ÷10 = A
    BMS_MAX_DISCHARGE_A  = 31026  # ÷10 = A
    # 31027 = inverter state code, 31028 = reserved
    BMS_CONNECT_STATE    = 31029  # 0 or 2 = not connected

    # -- Grid CT + Load (batch C: 31049–31054, 6 registers) --
    GRID_CT_POWER_H  = 31049  # high word of 32-bit grid CT power
    GRID_CT_POWER_L  = 31050  # low word  (raw 32-bit, scale −0.001 kW → negate for import+)
    LOAD_POWER       = 31054  # raw = W

    # -- Control (holding) --
    WORK_MODE        = 41000
    MIN_SOC          = 41009
    MIN_SOC_ON_GRID  = 41010

    # -- Remote control --
    REMOTE_ENABLE    = 44000  # 0=off, 1=on
    REMOTE_TIMEOUT   = 44001  # watchdog ms; inverter self-reverts if expired
    REMOTE_POWER     = 44002  # W (negative=charge from grid, positive=discharge to grid)


# BMS connect state values that indicate a valid connection
_BMS_VALID_STATES = {1, 3}  # anything other than 0 or 2


@dataclass
class InverterState:
    """Snapshot of current inverter/battery/grid state."""
    soc_pct: float          # battery state of charge (0-100%)
    soc_kwh: float          # derived from soc_pct and battery capacity
    battery_power_w: int    # positive = charging, negative = discharging
    battery_voltage_v: float
    pv_power_w: int         # total solar generation (PV1 + PV2)
    grid_power_w: int       # positive = importing, negative = exporting
    load_power_w: int       # home consumption
    battery_temp_c: float
    work_mode: WorkMode
    min_soc_pct: int
    bms_max_charge_w: int   # BMS-reported max charge power (W)
    bms_max_discharge_w: int  # BMS-reported max discharge power (W)


def _to_signed16(value: int) -> int:
    """Convert unsigned uint16 from Modbus to signed int16."""
    return value - 65536 if value > 32767 else value


def _to_signed32(high: int, low: int) -> int:
    """Combine two uint16 Modbus registers into a signed int32."""
    raw = (high << 16) | low
    return raw - 0x1_0000_0000 if raw >= 0x8000_0000 else raw


class FoxESSModbusClient:
    """Controls FoxESS KH series inverter via local Modbus TCP.

    KH-specific notes:
    - All monitoring registers are holding registers (FC 0x03), not input (FC 0x04).
    - KH only supports work modes 0-2. Force charge/discharge uses the remote
      control register block (44000-44002) with a watchdog timeout.
    - Export limit register 41012 is invalid on KH10 — no-export is expressed
      through the Back-up work mode (2) which suppresses solar export.

    Safety rules:
    - 250ms minimum between writes
    - Read-back verification after every work-mode write
    - Never set min SoC below 10%
    - Emergency self-use + remote control disable on communication failure
    """

    def __init__(self):
        self.ip = config.foxess.ip
        self.port = config.foxess.port
        self.slave_id = config.foxess.slave_id
        self.dry_run = config.system.dry_run
        self._client: ModbusTcpClient | None = None
        self._last_write_time = 0.0

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        if self._client and self._client.connected:
            return True
        self._client = ModbusTcpClient(
            host=self.ip,
            port=self.port,
            timeout=5,
            retries=3,
        )
        # Disable Nagle's algorithm for lower write latency
        try:
            import socket
            self._client.socket_params = {"TCP_NODELAY": socket.TCP_NODELAY}
        except Exception:
            pass
        connected = self._client.connect()
        if connected:
            logger.info("Connected to FoxESS KH inverter at %s:%s", self.ip, self.port)
        else:
            logger.error("Failed to connect to FoxESS inverter at %s:%s", self.ip, self.port)
        return connected

    def disconnect(self):
        if self._client:
            self._client.close()
            self._client = None

    # ------------------------------------------------------------------
    # State reading (3 batched FC03 reads)
    # ------------------------------------------------------------------

    def read_state(self) -> InverterState | None:
        """Read current inverter state using 3 batched holding-register reads.

        Reads are always performed even in dry-run mode so the optimizer
        works with real SoC, PV, load, and grid data.
        """
        if not self.connect():
            return None

        try:
            # Batch A: 31020–31029 — battery V/I/P, temp, SoC, BMS limits, BMS state
            bat = self._read_holding_registers(KHRegisters.BATTERY_VOLTAGE, 10)
            if bat is None:
                logger.error("Failed to read battery registers (31020-31029)")
                return None

            battery_voltage_v = bat[0] / 10.0                  # 31020
            battery_current_a = _to_signed16(bat[1]) / 10.0    # 31021
            battery_power_w   = _to_signed16(bat[2])            # 31022
            battery_temp_c    = _to_signed16(bat[3]) / 10.0    # 31023
            battery_soc       = bat[4]                           # 31024
            bms_max_charge_a  = bat[5] / 10.0                   # 31025
            bms_max_discharge_a = bat[6] / 10.0                 # 31026
            # bat[7] = inverter state (31027), bat[8] = reserved (31028)
            bms_connect_state = bat[9]                           # 31029

            if bms_connect_state not in _BMS_VALID_STATES:
                logger.warning(
                    "BMS not connected (state=%d) — SoC reading unreliable; skipping cycle",
                    bms_connect_state,
                )
                return None

            # Derive BMS max power from current limits × voltage
            bms_max_charge_w    = int(bms_max_charge_a    * battery_voltage_v)
            bms_max_discharge_w = int(bms_max_discharge_a * battery_voltage_v)

            # Batch B: 31002–31005 — PV1+PV2 power (and voltage/current, ignored)
            pv = self._read_holding_registers(KHRegisters.PV1_POWER, 4)
            pv1_power_w = _to_signed16(pv[0]) if pv else 0   # 31002
            pv2_power_w = _to_signed16(pv[2]) if pv else 0   # 31005 offset 3 from 31002

            # Batch C: 31049–31054 — grid CT (32-bit) + load
            grid_regs = self._read_holding_registers(KHRegisters.GRID_CT_POWER_H, 6)
            if grid_regs:
                # 31049-31050: 32-bit, scale −0.001 kW
                # raw > 0 → exporting → grid_power_w negative (matches convention)
                grid_raw = _to_signed32(grid_regs[0], grid_regs[1])
                grid_power_w = -grid_raw  # negate: import positive, export negative
                load_power_w = _to_signed16(grid_regs[5])  # 31054 (offset 5)
            else:
                grid_power_w = 0
                load_power_w = 0

            # Read control registers (work mode, min SoC) — individual reads
            hr = self._read_holding_registers(KHRegisters.WORK_MODE, 1)
            try:
                work_mode = WorkMode(hr[0]) if hr else WorkMode.SELF_USE
            except ValueError:
                work_mode = WorkMode.SELF_USE

            min_soc_r = self._read_holding_registers(KHRegisters.MIN_SOC, 1)
            min_soc = min_soc_r[0] if min_soc_r else 10

            soc_kwh = battery_soc / 100.0 * config.battery.capacity_kwh

            return InverterState(
                soc_pct=float(battery_soc),
                soc_kwh=soc_kwh,
                battery_power_w=battery_power_w,
                battery_voltage_v=battery_voltage_v,
                pv_power_w=pv1_power_w + pv2_power_w,
                grid_power_w=grid_power_w,
                load_power_w=load_power_w,
                battery_temp_c=battery_temp_c,
                work_mode=work_mode,
                min_soc_pct=min_soc,
                bms_max_charge_w=bms_max_charge_w,
                bms_max_discharge_w=bms_max_discharge_w,
            )

        except ModbusException as e:
            logger.error("Modbus read error: %s", e)
            return None

    # ------------------------------------------------------------------
    # Control — work mode
    # ------------------------------------------------------------------

    def set_work_mode(self, mode: WorkMode) -> bool:
        """Set the KH work mode (0=Self-Use, 1=Feed-In First, 2=Back-up)."""
        logger.info("Setting work mode to %s (%d)", mode.name, mode.value)
        if not self._write_register(KHRegisters.WORK_MODE, mode.value):
            return False

        if self.dry_run:
            return True

        # Read-back verification
        time.sleep(_VERIFY_DELAY)
        result = self._read_holding_registers(KHRegisters.WORK_MODE, 1)
        if result and result[0] == mode.value:
            logger.info("Work mode verified: %s", mode.name)
            return True
        logger.error("Work mode verification failed! Expected %d, got %s", mode.value, result)
        return False

    def set_min_soc(self, soc_pct: int) -> bool:
        """Set minimum SoC percentage (clamped to 10–100%)."""
        soc_pct = max(10, min(100, soc_pct))
        logger.info("Setting min SoC to %d%%", soc_pct)
        if not self._write_register(KHRegisters.MIN_SOC, soc_pct):
            return False
        self._write_register(KHRegisters.MIN_SOC_ON_GRID, soc_pct)
        return True

    # ------------------------------------------------------------------
    # Control — remote control (replaces FORCE_CHARGE / FORCE_DISCHARGE)
    # ------------------------------------------------------------------

    def set_remote_control(
        self,
        power_w: int,
        fallback_mode: WorkMode = WorkMode.BACKUP,
        timeout_ms: int = 600_000,
    ) -> bool:
        """Enable KH remote control mode to force a specific charge/discharge power.

        Args:
            power_w: Target power in watts.
                     Negative = charge from grid, positive = discharge to grid.
            fallback_mode: Work mode the inverter reverts to if the watchdog fires.
            timeout_ms: Watchdog timeout in ms. Default 600,000 (10 min) matches
                        the software watchdog in main.py.

        Writes 41000 (fallback), then 44001 (timeout), then 44002 (power), then 44000=1.
        """
        logger.info(
            "Remote control: power=%dW, fallback=%s, timeout=%dms",
            power_w, fallback_mode.name, timeout_ms,
        )
        # 1. Set fallback work mode (what inverter uses if watchdog fires)
        if not self._write_register(KHRegisters.WORK_MODE, fallback_mode.value):
            return False
        # 2. Set watchdog timeout
        if not self._write_register(KHRegisters.REMOTE_TIMEOUT, timeout_ms):
            return False
        # 3. Set target power
        if not self._write_register(KHRegisters.REMOTE_POWER, power_w):
            return False
        # 4. Enable remote control
        if not self._write_register(KHRegisters.REMOTE_ENABLE, 1):
            return False
        return True

    def disable_remote_control(self) -> bool:
        """Disable remote control and return inverter to normal scheduling."""
        logger.info("Disabling remote control")
        return self._write_register(KHRegisters.REMOTE_ENABLE, 0)

    def emergency_self_use(self):
        """Force inverter back to self-use mode. Called on errors or watchdog timeout."""
        logger.warning("EMERGENCY: Disabling remote control and forcing self-use mode")
        try:
            if self.connect():
                self._write_register(KHRegisters.REMOTE_ENABLE, 0)
                self._write_register(KHRegisters.WORK_MODE, WorkMode.SELF_USE.value)
                self._write_register(KHRegisters.MIN_SOC, 10)
        except Exception as e:
            logger.error("Emergency self-use failed: %s", e)

    # ------------------------------------------------------------------
    # Low-level Modbus operations
    # ------------------------------------------------------------------

    def _read_holding_registers(self, address: int, count: int) -> list[int] | None:
        try:
            result = self._client.read_holding_registers(
                address=address, count=count, slave=self.slave_id
            )
            if result.isError():
                logger.error("Read holding register %d (count=%d) error: %s",
                             address, count, result)
                return None
            return result.registers
        except Exception as e:
            logger.error("Read holding register %d (count=%d) exception: %s",
                         address, count, e)
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
