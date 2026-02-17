# FoxESS KH Series — Modbus Register Reference

Source: `KH.Series.Modbus.pdf` (FoxESS official documentation)

Connection: Modbus TCP, default port 502, slave ID 247

## Data Types

| Name        | Description                    |
|-------------|--------------------------------|
| U16         | Unsigned integer (16 bits)     |
| U32         | Unsigned integer (32 bits)     |
| I16         | Signed integer (16 bits)       |
| I32         | Signed integer (32 bits)       |
| STR         | Character string               |
| Bitfield16  | 16-bit data expressed by bit   |
| Bitfield32  | 32-bit data expressed by bit   |
| RO          | Read only                      |
| RW          | Read and write                 |
| WO          | Write only                     |

## Gain

Where `Gain` is listed (e.g. 10), the raw register value must be divided by the
gain to get the real value.  Example: `PV1 voltage` at gain 10 means a raw
register value of 2345 = 234.5 V.

---

## 1. Device Information (RO) — Address 30000+

| Signal                  | RW | Type | Unit | Gain | Address | Qty |
|-------------------------|----|------|------|------|---------|-----|
| Model                   | RO | STR  | —    | 1    | 30000   | 16  |
| Firmware Master         | RO | U16  | —    | 1    | 30016   | 1   |
| Firmware Slave          | RO | U16  | —    | 1    | 30017   | 1   |
| Firmware Manager        | RO | U16  | —    | 1    | 30018   | 1   |
| Firmware Battery Master | RO | U16  | —    | 1    | 30019   | 1   |
| Firmware Battery Slave1 | RO | U16  | —    | 1    | 30020   | 1   |
| Firmware Battery Slave2 | RO | U16  | —    | 1    | 30021   | 1   |
| Firmware Battery Slave3 | RO | U16  | —    | 1    | 30022   | 1   |
| Firmware Battery Slave4 | RO | U16  | —    | 1    | 30023   | 1   |
| Firmware Battery Slave5 | RO | U16  | —    | 1    | 30024   | 1   |
| Firmware Battery Slave6 | RO | U16  | —    | 1    | 30025   | 1   |
| Firmware Battery Slave7 | RO | U16  | —    | 1    | 30026   | 1   |
| Firmware Battery Slave8 | RO | U16  | —    | 1    | 30027   | 1   |
| BmsMasterSN             | RO | STR  | —    | 1    | 30028   | 15  |
| BmsSlave1SN             | RO | STR  | —    | 1    | 30043   | 15  |
| BmsSlave2SN             | RO | STR  | —    | 1    | 30058   | 15  |
| BmsSlave3SN             | RO | STR  | —    | 1    | 30073   | 15  |
| BmsSlave4SN             | RO | STR  | —    | 1    | 30088   | 15  |
| BmsSlave5SN             | RO | STR  | —    | 1    | 30103   | 15  |
| BmsSlave6SN             | RO | STR  | —    | 1    | 30118   | 15  |
| BmsSlave7SN             | RO | STR  | —    | 1    | 30133   | 15  |
| BmsSlave8SN             | RO | STR  | —    | 1    | 30148   | 15  |

---

## 2. Live Measurements (RO) — Address 31000+

### PV Strings

| Signal      | RW | Type | Unit | Gain | Address | Qty | Notes |
|-------------|----|------|------|------|---------|-----|-------|
| PV1 voltage | RO | I16  | V    | 10   | 31000   | 1   |       |
| PV1 current | RO | I16  | A    | 10   | 31001   | 1   |       |
| PV1 power   | RO | I16  | W    | 1    | 31002   | 1   |       |
| PV2 voltage | RO | I16  | V    | 10   | 31003   | 1   |       |
| PV2 current | RO | I16  | A    | 10   | 31004   | 1   |       |
| PV2 power   | RO | I16  | W    | 1    | 31005   | 1   |       |
| PV3 voltage | RO | I16  | V    | 10   | 31039   | 1   |       |
| PV3 current | RO | I16  | A    | 10   | 31040   | 1   |       |
| PV3 power   | RO | I16  | W    | 1    | 31041   | 1   |       |
| PV4 voltage | RO | I16  | V    | 10   | 31042   | 1   |       |
| PV4 current | RO | I16  | A    | 10   | 31043   | 1   |       |
| PV4 power   | RO | I16  | W    | 1    | 31044   | 1   |       |

### Grid & Inverter

| Signal          | RW | Type | Unit | Gain | Address | Qty | Notes |
|-----------------|----|------|------|------|---------|-----|-------|
| Grid voltage    | RO | U16  | V    | 10   | 31006   | 1   |       |
| Inv current     | RO | I16  | A    | 10   | 31007   | 1   | Inverter output current |
| Inv power       | RO | I16  | W    | 1    | 31008   | 1   | Inverter output power |
| Grid frequency  | RO | U16  | Hz   | 100  | 31009   | 1   |       |
| Eps voltage     | RO | U16  | V    | 10   | 31010   | 1   | Emergency power supply |
| Eps current     | RO | I16  | A    | 10   | 31011   | 1   |       |
| Eps power       | RO | I16  | W    | 1    | 31012   | 1   |       |
| Eps frequency   | RO | U16  | Hz   | 100  | 31013   | 1   |       |

### Meters & Load

| Signal              | RW | Type | Unit | Gain | Address | Qty | Notes |
|---------------------|----|------|------|------|---------|-----|-------|
| Meter1 power        | RO | I16  | W    | 1    | 31014   | 1   | CT clamp / meter — positive=import, negative=export |
| Meter2 power        | RO | I16  | W    | 1    | 31015   | 1   |       |
| Load power          | RO | I16  | W    | 1    | 31016   | 1   | Home consumption |

### Temperatures

| Signal               | RW | Type | Unit | Gain | Address | Qty | Notes |
|----------------------|----|------|------|------|---------|-----|-------|
| Inverter temperature | RO | I16  | C    | 10   | 31018   | 1   |       |
| Internal temperature | RO | I16  | C    | 10   | 31019   | 1   |       |

### Battery

| Signal                     | RW | Type | Unit | Gain | Address | Qty | Notes |
|----------------------------|----|------|------|------|---------|-----|-------|
| Battery voltage            | RO | I16  | V    | 10   | 31020   | 1   |       |
| Battery current            | RO | I16  | A    | 10   | 31021   | 1   | Positive=charging, negative=discharging |
| Battery power              | RO | I16  | W    | 1    | 31022   | 1   | Positive=charging, negative=discharging |
| Battery temperature        | RO | I16  | C    | 10   | 31023   | 1   |       |
| SoC                        | RO | U16  | %    | 1    | 31024   | 1   | 0–100 |
| Maximum charge current     | RO | U16  | A    | 10   | 31025   | 1   | BMS-reported max |
| Maximum discharge current  | RO | U16  | A    | 10   | 31026   | 1   | BMS-reported max |

### Status & Faults

| Signal               | RW | Type       | Unit | Gain | Address | Qty | Values |
|----------------------|----|------------|------|------|---------|-----|--------|
| Inverter state       | RO | U16        | —    | 1    | 31027   | 1   | 0: Self-Test, 1: WaitState, 2: CheckState, 3: Normal, 4: EpsState, 5: FaultState, 6: Permanent FaultState, 8: FlashState |
| BMS connect state    | RO | U16        | —    | 1    | 31028   | 1   |        |
| Meter1 connect state | RO | U16        | —    | 1    | 31029   | 1   |        |
| Meter2 connect state | RO | U16        | —    | 1    | 31030   | 1   |        |
| Fault 1              | RO | Bitfield16 | —    | 1    | 31031   | 1   | See fault table below |
| Fault 2              | RO | Bitfield16 | —    | 1    | 31032   | 1   | See fault table below |
| Fault 3              | RO | Bitfield16 | —    | 1    | 31033   | 1   | See fault table below |
| Fault 4              | RO | Bitfield16 | —    | 1    | 31034   | 1   | See fault table below |
| Fault 5              | RO | Bitfield16 | —    | 1    | 31035   | 1   | See fault table below |
| Fault 6              | RO | Bitfield16 | —    | 1    | 31036   | 1   | See fault table below |
| Fault 7              | RO | Bitfield16 | —    | 1    | 31037   | 1   | See fault table below |
| Fault 8              | RO | Bitfield16 | —    | 1    | 31038   | 1   | See fault table below |

---

## 3. Energy Counters (RO) — Address 32000+

| Signal                    | RW | Type | Unit | Gain | Address | Qty | Notes |
|---------------------------|----|------|------|------|---------|-----|-------|
| Total PV energy           | RO | U32  | kWh  | 10   | 32000   | 2   | Lifetime |
| Today PV energy           | RO | U16  | kWh  | 10   | 32002   | 1   |       |
| Total charge energy       | RO | U32  | kWh  | 10   | 32003   | 2   | Lifetime battery charge |
| Today charge energy       | RO | U16  | kWh  | 10   | 32005   | 1   |       |
| Total discharge energy    | RO | U32  | kWh  | 10   | 32006   | 2   | Lifetime battery discharge |
| Today discharge energy    | RO | U16  | kWh  | 10   | 32008   | 1   |       |
| Total feed-in energy      | RO | U32  | kWh  | 10   | 32009   | 2   | Lifetime grid export |
| Today feed-in energy      | RO | U16  | kWh  | 10   | 32011   | 1   |       |
| Total consumption energy  | RO | U32  | kWh  | 10   | 32012   | 2   | Lifetime grid import |
| Today consumption energy  | RO | U16  | kWh  | 10   | 32014   | 1   |       |
| Total output energy       | RO | U32  | kWh  | 10   | 32015   | 2   | Lifetime inverter output |
| Today output energy       | RO | U16  | kWh  | 10   | 32017   | 1   |       |
| Total input energy        | RO | U32  | kWh  | 10   | 32018   | 2   | Lifetime inverter input |
| Today input energy        | RO | U16  | kWh  | 10   | 32020   | 1   |       |
| Total load energy         | RO | U32  | kWh  | 10   | 32021   | 2   | Lifetime home load |
| Today load energy         | RO | U16  | kWh  | 10   | 32023   | 1   |       |

---

## 4. Real-Time Clock (RW) — Address 40000+

| Signal     | RW | Type | Unit | Gain | Address | Qty | Range |
|------------|----|------|------|------|---------|-----|-------|
| RTC year   | RW | U16  | —    | 1    | 40000   | 1   | 0–99  |
| RTC month  | RW | U16  | —    | 1    | 40001   | 1   | 1–12  |
| RTC day    | RW | U16  | —    | 1    | 40002   | 1   | 1–31  |
| RTC hour   | RW | U16  | —    | 1    | 40003   | 1   | 0–23  |
| RTC minute | RW | U16  | —    | 1    | 40004   | 1   | 0–59  |
| RTC second | RW | U16  | —    | 1    | 40005   | 1   | 0–59  |

---

## 5. Control Registers (RW) — Address 41000+

| Signal                        | RW | Type | Unit | Gain | Address | Qty | Notes |
|-------------------------------|----|------|------|------|---------|-----|-------|
| Work mode                     | RW | U16  | —    | 1    | 41000   | 1   | 0: Self-Use, 1: Feed-in First, 2: Backup, 3: Force Charge, 4: Force Discharge |
| Maximum set charge current    | RW | U16  | A    | 10   | 41007   | 1   | Divide raw value by 10 for amps |
| Maximum set discharge current | RW | U16  | A    | 10   | 41008   | 1   | Divide raw value by 10 for amps |
| Minimum SoC                   | RW | U16  | %    | 1    | 41009   | 1   | Battery won't discharge below this |
| Maximum SoC                   | RW | U16  | %    | 1    | 41010   | 1   | Battery won't charge above this |
| Minimum SoC (On Grid)         | RW | U16  | %    | 1    | 41011   | 1   | Min SoC when grid is available |
| **Export limit**               | RW | U16  | **W** | 1   | **41012** | 1 | Grid export power cap in watts. 0 = no export. |
| Sys on/off                    | RW | U16  | —    | 1    | 41013   | 1   |       |
| Eps frequency select          | RW | U16  | —    | —    | 41014   | 1   |       |
| Eps output                    | RW | U16  | —    | —    | 41015   | 1   |       |
| Grounding                     | RW | U16  | —    | 1    | 41016   | 1   | 0: Disable, 1: Enable |

---

## 6. Remote Power Control (RW) — Address 44000+

| Signal                          | RW | Type | Unit | Gain | Address | Qty | Notes |
|---------------------------------|----|------|------|------|---------|-----|-------|
| Remote power control enable     | RW | U16  | —    | 1    | 44000   | 1   |       |
| Remote power control timeout    | RW | U16  | s    | 1    | 44001   | 1   | Seconds before reverting |
| Active power command            | RW | I16  | W    | 1    | 44002   | 1   | Positive=charge, negative=discharge |
| Reactive power command          | RW | I16  | Var  | 1    | 44003   | 1   |       |

---

## 7. Fault Bit Definitions

### Fault 1 (register 31031)

| Bit | Name               | Description                    |
|-----|--------------------|--------------------------------|
| 0   | GridLostFault      | Grid lost                      |
| 1   | GridVoltFault      | Grid voltage fault             |
| 2   | GridFreqFault      | Grid frequency fault           |
| 3   | Grid10minVoltFault | Grid 10-min voltage fault      |
| 4   | EpsVoltFault       | EPS voltage fault              |
| 5   | SwInvCurFault      | Inverter current fault         |
| 6   | DciFault           | DC injection fault             |
| 7   | TBD                |                                |
| 8   | EpsVolFault        | EPS voltage fault (alt)        |
| 9   | SwBusVoltFault     | Bus voltage fault              |
| 10  | BatOvpFault        | Battery over-voltage fault     |
| 11  | TBD                |                                |
| 12  | IsoFault           | Isolation fault                |
| 13  | ResCurFault        | Residual current fault         |
| 14  | PvVoltFault        | PV voltage fault               |
| 15  | DirectArcFault     | Arc fault                      |

### Fault 2 (register 31032)

| Bit | Name              | Description                      |
|-----|-------------------|----------------------------------|
| 0   | TempFault         | Temperature fault                |
| 1   | GroundConnFault   | Ground connection fault          |
| 4   | BatPowerLowFault  | Battery power low                |
| 7   | EpsOverLoad       | EPS overload                     |
| 8   | SciFault          | Master-Manager comm fault        |
| 9   | SpiCommFault      | Master-Slave comm fault          |
| 10  | BmsLostFault      | Master-BMS comm fault            |
| 12  | ItalySelftest     | Italy self-test                  |
| 14  | CanNodeNum        | CAN node number fault            |

### Fault 4 (register 31034)

| Bit | Name                 | Description                    |
|-----|----------------------|--------------------------------|
| 0   | SampleCircuitOffset  | Sample circuit offset          |
| 1   | RCDevice             | RC device fault                |
| 2   | EEpromFault          | EEPROM fault                   |
| 3   | PvNegCurt            | PV negative current            |
| 4   | BatRly_KeepOpen      | Battery relay stuck open       |
| 5   | BatRly_KeepClose     | Battery relay stuck closed     |
| 6   | BatBuck              | Battery BUCK circuit fault     |
| 7   | BatBoost             | Battery BOOST circuit fault    |
| 9   | ShortEpsLoad         | EPS short circuit              |
| 10  | BatConnDirFault      | Battery connection direction   |
| 11  | MainRelayOpenFault   | Main relay open fault          |
| 12  | MainRelayS1KeepClosed| S1 relay stuck closed          |
| 13  | MainRelayS2KeepClosed| S2 relay stuck closed          |
| 14  | MainRelayM1KeepClosed| M1 relay stuck closed          |
| 15  | MainRelayM2KeepClosed| M2 relay stuck closed          |

### Fault 6 (register 31036)

| Bit | Name              | Description              |
|-----|-------------------|--------------------------|
| 0   | ArmEepromFault    | Manager EEPROM fault     |
| 1   | MeterLostFault    | Meter lost               |
| 2   | CtLostFault       | CT clamp lost            |

### Fault 7 (register 31037) — BMS Faults

| Bit | Name                  | Description                    |
|-----|-----------------------|--------------------------------|
| 0   | BmsExternalFault      | BMS external fault             |
| 1   | BmsInternalFault      | BMS internal fault             |
| 2   | BmsVoltHighFault      | BMS voltage too high           |
| 3   | BmsVoltLowFault       | BMS voltage too low            |
| 4   | BmsChgCurHighFault    | BMS charge current too high    |
| 5   | BmsDischgCurHighFault | BMS discharge current too high |
| 6   | BmsTempHighFault      | BMS temperature too high       |
| 7   | BmsTempLowFault       | BMS temperature too low        |
| 8   | BmsCellImbalance      | BMS cell imbalance             |
| 9   | BmsHardwareProtect    | BMS hardware protection        |
| 10  | BmsCircuitFault       | BMS circuit fault              |
| 11  | BmsInsulationFault    | BMS insulation fault           |
| 12  | BmsVoltSensorFault    | BMS voltage sensor fault       |
| 13  | BmsTempSensorFault    | BMS temperature sensor fault   |
| 14  | BmsCurSensorFault     | BMS current sensor fault       |
| 15  | BmsRelayFault         | BMS relay fault                |

### Fault 8 (register 31038) — BMS Compatibility

| Bit | Name                             | Description                         |
|-----|----------------------------------|-------------------------------------|
| 0   | BmsTypeUnmatch                   | BMS type mismatch                   |
| 1   | BmsVersionUnmatch                | BMS version mismatch                |
| 2   | BmsManufacturerUnmatch           | BMS manufacturer mismatch           |
| 3   | BmsSwHwUnmatch                   | BMS software/hardware mismatch      |
| 4   | BmsMSUnmatch                     | BMS master/slave mismatch           |
| 5   | BmsChgReqNoReply                 | BMS charge request no reply         |
| 6   | BmsSupplyFault                   | BMS supply fault                    |
| 8   | BmsSelfChkFault                  | BMS self-check fault                |
| 9   | BmsCellTempDiffFault             | BMS cell temperature diff fault     |
| 10  | BmsCellVoltBreakLineFault        | BMS cell voltage break line fault   |
| 11  | BmsSelfChkSysVoltMisMatchFault   | BMS system voltage mismatch         |
| 12  | BmsPreChgFault                   | BMS pre-charge fault                |
| 13  | BmsSelfChkHvbFault               | BMS HVB self-check fault            |
| 14  | BmsSelfChkPackCurFault           | BMS pack current self-check fault   |
| 15  | BmsSelfChkSysMismatchFault       | BMS system mismatch self-check      |

---

## 8. Code Mapping — Registers Used in This Project

The `foxess/modbus_client.py` module uses a subset of these registers.  Below is
the mapping between register addresses and the code constants.

### Input Registers (read via `read_input_registers`)

Note: The current code uses address offsets from the H1 series (11000+).
The KH series uses 31000+ addresses.  Verify which addressing scheme your
pymodbus client and inverter firmware expect.

| Code Constant                     | Code Address | KH PDF Address | Signal       |
|-----------------------------------|-------------|----------------|--------------|
| `InputRegisters.PV1_POWER`        | 11002       | 31002          | PV1 power    |
| `InputRegisters.PV2_POWER`        | 11005       | 31005          | PV2 power    |
| `InputRegisters.GRID_CT_POWER`    | 11021       | 31014          | Meter1 power |
| `InputRegisters.LOAD_POWER`       | 11023       | 31016          | Load power   |
| `InputRegisters.BATTERY_VOLTAGE`  | 11033       | 31020          | Bat voltage  |
| `InputRegisters.BATTERY_POWER`    | 11034       | 31022          | Bat power    |
| `InputRegisters.BATTERY_CURRENT`  | 11035       | 31021          | Bat current  |
| `InputRegisters.BATTERY_SOC`      | 11036       | 31024          | SoC          |
| `InputRegisters.BATTERY_TEMP`     | 11038       | 31023          | Bat temp     |

### Holding Registers (read/write via `read_holding_registers` / `write_register`)

| Code Constant                          | Address | Signal                        |
|----------------------------------------|---------|-------------------------------|
| `HoldingRegisters.WORK_MODE`           | 41000   | Work mode                     |
| `HoldingRegisters.TIME1_ENABLE`        | 41001   | Time period 1 enable          |
| `HoldingRegisters.TIME1_START_HOUR`    | 41002   | Time period 1 start hour      |
| `HoldingRegisters.TIME1_START_MIN`     | 41003   | Time period 1 start minute    |
| `HoldingRegisters.TIME1_END_HOUR`      | 41004   | Time period 1 end hour        |
| `HoldingRegisters.TIME1_END_MIN`       | 41005   | Time period 1 end minute      |
| `HoldingRegisters.TIME1_CHARGE_POWER`  | 41006   | Time period 1 charge power    |
| `HoldingRegisters.TIME1_DISCHARGE_POWER`| 41007  | Max set charge current (KH)   |
| `HoldingRegisters.MIN_SOC`             | 41009   | Minimum SoC                   |
| `HoldingRegisters.MIN_SOC_ON_GRID`     | 41010   | Maximum SoC (KH) / Min SoC on grid (H1) |
| `HoldingRegisters.EXPORT_LIMIT`        | 41012   | Grid export power limit (W)   |
