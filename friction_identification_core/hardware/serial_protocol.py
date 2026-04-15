from __future__ import annotations

"""UART frame packing and incremental parsing helpers for the real robot."""

import struct
from dataclasses import dataclass

import numpy as np


RECV_FRAME_HEAD = 0xA5
RECV_FRAME_FORMAT = "<BBBfffff"
RECV_FRAME_STRUCT = struct.Struct(RECV_FRAME_FORMAT)
RECV_FRAME_SIZE = RECV_FRAME_STRUCT.size

SEND_FRAME_HEAD = b"\xAA\x55"
SEND_FRAME_TAIL = b"\x55\xAA"
SEND_PAYLOAD_STRUCT = struct.Struct("<7f")
SEND_FRAME_STRUCT = struct.Struct("<2s7fB2s")
SEND_FRAME_SIZE = SEND_FRAME_STRUCT.size


def calculate_xor_checksum(data: bytes) -> int:
    """Compute the 8-bit XOR checksum expected by the UART protocol."""

    checksum = 0
    for byte in data:
        checksum ^= byte
    return checksum & 0xFF


@dataclass(frozen=True)
class JointFeedbackFrame:
    """Decoded state feedback for one motor in the 7-axis chain."""

    motor_id: int
    state: int
    position: float
    velocity: float
    torque: float
    mos_temperature: float
    coil_temperature: float


class TorqueCommandFramePacker:
    """Pack mode-1 UART frames that carry 7-axis torque commands."""

    def __init__(self) -> None:
        self._frame = bytearray(SEND_FRAME_SIZE)
        self._checksum_offset = 2 + SEND_PAYLOAD_STRUCT.size

    def pack(self, torque_command: np.ndarray) -> bytes:
        torques = np.asarray(torque_command, dtype=np.float32).reshape(-1)
        if torques.size != 7:
            raise ValueError("torque_command must contain exactly 7 floats.")

        self._frame[0:2] = SEND_FRAME_HEAD
        SEND_PAYLOAD_STRUCT.pack_into(self._frame, 2, *[float(value) for value in torques])
        self._frame[self._checksum_offset] = calculate_xor_checksum(self._frame[: self._checksum_offset])
        self._frame[self._checksum_offset + 1 : self._checksum_offset + 3] = SEND_FRAME_TAIL
        return bytes(self._frame)


class SerialFrameReader:
    """Incrementally decode motor feedback frames from the UART byte stream."""

    def __init__(self) -> None:
        self._buffer = bytearray()

    def read_available(self, ser) -> int:
        bytes_waiting = ser.in_waiting
        if bytes_waiting <= 0:
            return 0
        chunk = ser.read(bytes_waiting)
        if chunk:
            self._buffer.extend(chunk)
        return len(chunk)

    def has_complete_frame(self) -> bool:
        return len(self._buffer) >= RECV_FRAME_SIZE

    def pop_frame(self) -> JointFeedbackFrame | None:
        while True:
            if len(self._buffer) < RECV_FRAME_SIZE:
                return None

            head_index = self._buffer.find(RECV_FRAME_HEAD)
            if head_index < 0:
                self._buffer.clear()
                return None
            if head_index > 0:
                del self._buffer[:head_index]
                if len(self._buffer) < RECV_FRAME_SIZE:
                    return None

            try:
                parsed = RECV_FRAME_STRUCT.unpack_from(self._buffer)
            except struct.error:
                return None

            del self._buffer[:RECV_FRAME_SIZE]
            return JointFeedbackFrame(
                motor_id=int(parsed[1]),
                state=int(parsed[2]),
                position=float(parsed[3]),
                velocity=float(parsed[4]),
                torque=float(parsed[5]),
                mos_temperature=float(parsed[6]),
                coil_temperature=float(parsed[7]),
            )
