from __future__ import annotations

"""UART frame packing and incremental parsing helpers for the real robot."""

import struct
from dataclasses import dataclass

import numpy as np


RECV_FRAME_HEAD = 0xA5
# head, motor_id, state, pos, vel, tor, Tmos
RECV_FRAME_FORMAT = "<BBBffff"
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
    # The current UART feedback frame does not carry coil temperature.
    # Keep a NaN placeholder so downstream storage/visualization stays compatible.
    coil_temperature: float = float("nan")


class TorqueCommandFramePacker:
    """Pack mode-1 UART frames that carry 7-axis torque commands."""

    def __init__(self) -> None:
        self._frame = bytearray(SEND_FRAME_SIZE)
        self._header_size = len(SEND_FRAME_HEAD)
        self._payload_offset = self._header_size
        self._payload_size = SEND_PAYLOAD_STRUCT.size
        self._checksum_offset = self._header_size + SEND_PAYLOAD_STRUCT.size

    def pack(self, torque_command: np.ndarray) -> bytes:
        torques = np.asarray(torque_command, dtype=np.float32).reshape(-1)
        if torques.size != 7:
            raise ValueError("torque_command must contain exactly 7 floats.")

        self._frame[0:self._header_size] = SEND_FRAME_HEAD
        SEND_PAYLOAD_STRUCT.pack_into(self._frame, self._payload_offset, *[float(value) for value in torques])
        checksum_end = self._payload_offset + self._payload_size
        # The device-side mode-1 protocol validates XOR over frame head + payload.
        self._frame[self._checksum_offset] = calculate_xor_checksum(
            self._frame[0:checksum_end]
        )
        self._frame[self._checksum_offset + 1 : self._checksum_offset + 3] = SEND_FRAME_TAIL
        return bytes(self._frame)


class SerialFrameReader:
    """Incrementally decode motor feedback frames from the UART byte stream."""

    def __init__(self, *, max_motor_id: int = 7) -> None:
        self._buffer = bytearray()
        self._max_motor_id = max(int(max_motor_id), 1)

    def _is_valid_candidate(self, parsed: tuple[object, ...], *, has_following_frame: bool) -> bool:
        if int(parsed[0]) != RECV_FRAME_HEAD:
            return False

        motor_id = int(parsed[1])
        if not 1 <= motor_id <= self._max_motor_id:
            return False

        feedback_values = np.asarray(parsed[3:], dtype=np.float32)
        if not np.all(np.isfinite(feedback_values)):
            return False

        # When two full frames are already buffered, a valid candidate should align
        # with the next frame boundary as well.
        if has_following_frame and self._buffer[RECV_FRAME_SIZE] != RECV_FRAME_HEAD:
            return False

        return True

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

            if not self._is_valid_candidate(
                parsed,
                has_following_frame=len(self._buffer) >= (RECV_FRAME_SIZE * 2),
            ):
                del self._buffer[0]
                continue

            del self._buffer[:RECV_FRAME_SIZE]
            return JointFeedbackFrame(
                motor_id=int(parsed[1]),
                state=int(parsed[2]),
                position=float(parsed[3]),
                velocity=float(parsed[4]),
                torque=float(parsed[5]),
                mos_temperature=float(parsed[6]),
            )


__all__ = [
    "JointFeedbackFrame",
    "RECV_FRAME_SIZE",
    "SEND_FRAME_SIZE",
    "SerialFrameReader",
    "TorqueCommandFramePacker",
    "calculate_xor_checksum",
]
