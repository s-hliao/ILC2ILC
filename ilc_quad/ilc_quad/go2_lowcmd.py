"""Go2 low-level message helpers for go2_bridge: leg order and the LowCmd CRC.

Kept free of ROS and of unitree_go so they can be tested anywhere; the bridge
passes in unitree_go/LowCmd messages, which only need the attributes read here.

Leg order. Unitree's motor array runs FR, FL, RR, RL (hip, thigh, calf each), the
same order as Go1's MJCF; this package's canonical order is FL, FR, RL, RR. Joint
signs agree -- Menagerie's Go2 is converted from Unitree's URDF.

CRC. Go2 drops a LowCmd whose `crc` does not match. The value is Unitree's
crc32_core over the C struct `LowCmd_` (812 bytes, little-endian, naturally
aligned) minus its trailing crc word: CRC-32 with polynomial 0x04C11DB7, init
0xFFFFFFFF, no reflection and no final xor, fed one 32-bit word at a time from the
most significant bit. That is CRC-32/MPEG-2 over each word's big-endian bytes,
which is how it is computed here, with a table rather than bit by bit so it keeps
up with 500 Hz in Python. The struct layout follows unitree_sdk2's LowCmd_ (as
packed by unitree_sdk2_python's crc.py): MotorCmd_ is mode, q, dq, tau, kp, kd,
reserve[3].
"""

from __future__ import annotations

import struct

import numpy as np

from .sim_quad_model import CANONICAL_LEGS, NU

UNITREE_LEGS = ("FR", "FL", "RR", "RL")

# canonical joint i lives in Unitree motor slot CANONICAL_TO_MOTOR[i]
CANONICAL_TO_MOTOR = np.array(
    [3 * UNITREE_LEGS.index(leg) + j for leg in CANONICAL_LEGS for j in range(3)], dtype=int
)
# Unitree foot_force[k] is leg UNITREE_LEGS[k]; canonical leg i reads FOOT_FROM_UNITREE[i]
FOOT_FROM_UNITREE = np.array([UNITREE_LEGS.index(leg) for leg in CANONICAL_LEGS], dtype=int)

N_MOTOR_SLOTS = 20              # LowCmd carries 20 motor slots; Go2 uses the first 12
HEAD = (0xFE, 0xEF)
LEVEL_LOWLEVEL = 0xFF
MODE_FOC = 0x01                 # servo (PMSM) mode; 0x00 is off
POS_STOP_F = 2.146e9            # "no position target" sentinels from Unitree's examples
VEL_STOP_F = 16000.0

_LOWCMD_FMT = "<4B4IH2x" + "B3x5f3I" * N_MOTOR_SLOTS + "4B" + "55Bx2I"
LOWCMD_SIZE = struct.calcsize(_LOWCMD_FMT)          # 812, sizeof(LowCmd_)


def _mpeg2_table():
    table = []
    for byte in range(256):
        crc = byte << 24
        for _ in range(8):
            crc = ((crc << 1) ^ 0x04C11DB7) if crc & 0x80000000 else (crc << 1)
            crc &= 0xFFFFFFFF
        table.append(crc)
    return table


_TABLE = _mpeg2_table()


def crc32_core(words) -> int:
    """Unitree's crc32_core, bit by bit, as in the C source. Reference for tests only."""
    crc = 0xFFFFFFFF
    for data in words:
        bit = 1 << 31
        for _ in range(32):
            crc = (((crc << 1) ^ 0x04C11DB7) if crc & 0x80000000 else (crc << 1)) & 0xFFFFFFFF
            if data & bit:
                crc ^= 0x04C11DB7
            bit >>= 1
    return crc


def crc32_words_fast(packed: bytes) -> int:
    """crc32_core over little-endian 32-bit words, via the MPEG-2 byte table."""
    crc = 0xFFFFFFFF
    for i in range(0, len(packed), 4):
        for byte in (packed[i + 3], packed[i + 2], packed[i + 1], packed[i]):
            crc = ((crc << 8) & 0xFFFFFFFF) ^ _TABLE[((crc >> 24) ^ byte) & 0xFF]
    return crc


def pack_lowcmd(cmd) -> bytes:
    """The C struct bytes of a LowCmd message (crc field included, as it stands)."""
    values = [*cmd.head, cmd.level_flag, cmd.frame_reserve, *cmd.sn, *cmd.version, cmd.bandwidth]
    for m in cmd.motor_cmd:
        values += [m.mode, m.q, m.dq, m.tau, m.kp, m.kd, *m.reserve]
    values += [cmd.bms_cmd.off, *cmd.bms_cmd.reserve, *cmd.wireless_remote, *cmd.led,
               *cmd.fan, cmd.gpio, cmd.reserve, cmd.crc]
    return struct.pack(_LOWCMD_FMT, *values)


def lowcmd_crc(cmd) -> int:
    """The value Go2 expects in cmd.crc: crc32_core over every word but the crc itself."""
    return crc32_words_fast(pack_lowcmd(cmd)[:-4])


def fill_lowcmd(cmd, q, dq, kp, kd, tau) -> None:
    """Write canonical (12,) targets into a LowCmd: header, 12 motors in FOC mode, crc.

    Slots 12..19 are left in FOC mode with the stop sentinels and zero gains, as
    Unitree's examples initialize them.
    """
    cmd.head = list(HEAD)
    cmd.level_flag = LEVEL_LOWLEVEL
    cmd.gpio = 0
    for slot in range(N_MOTOR_SLOTS):
        m = cmd.motor_cmd[slot]
        m.mode = MODE_FOC
        m.q, m.dq, m.kp, m.kd, m.tau = POS_STOP_F, VEL_STOP_F, 0.0, 0.0, 0.0
    for i in range(NU):
        m = cmd.motor_cmd[int(CANONICAL_TO_MOTOR[i])]
        m.q, m.dq = float(q[i]), float(dq[i])
        m.kp, m.kd, m.tau = float(kp[i]), float(kd[i]), float(tau[i])
    cmd.crc = lowcmd_crc(cmd)
