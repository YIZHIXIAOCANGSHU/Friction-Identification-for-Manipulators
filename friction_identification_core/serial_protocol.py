from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np


RECV_FRAME_HEAD = 0xA5
RECV_FRAME_FORMAT = "<BBBffff"
RECV_FRAME_STRUCT = struct.Struct(RECV_FRAME_FORMAT)
RECV_FRAME_SIZE = RECV_FRAME_STRUCT.size

COMMAND_FRAME_HEAD = b"\xAA\x55"
COMMAND_FRAME_TAIL = b"\x55\xAA"
COMMAND_PAYLOAD_STRUCT = struct.Struct("<7f")
COMMAND_FRAME_SIZE = len(COMMAND_FRAME_HEAD) + COMMAND_PAYLOAD_STRUCT.size + 1 + len(COMMAND_FRAME_TAIL)


def calculate_xor_checksum(data: bytes) -> int:
    checksum = 0
    for value in data:
        checksum ^= int(value)
    return checksum & 0xFF


@dataclass(frozen=True)
class FeedbackFrame:
    motor_id: int
    state: int
    position: float
    velocity: float
    torque: float
    mos_temperature: float


class SerialFrameParser:
    def __init__(self, *, max_motor_id: int = 7) -> None:
        self._buffer = bytearray()
        self._max_motor_id = max(int(max_motor_id), 1)

    def feed(self, chunk: bytes) -> None:
        if chunk:
            self._buffer.extend(chunk)

    def _is_valid_candidate(self, parsed: tuple[object, ...], *, has_following_frame: bool) -> bool:
        if int(parsed[0]) != RECV_FRAME_HEAD:
            return False
        motor_id = int(parsed[1])
        if not 1 <= motor_id <= self._max_motor_id:
            return False
        values = np.asarray(parsed[3:], dtype=np.float32)
        if not np.all(np.isfinite(values)):
            return False
        if has_following_frame and self._buffer[RECV_FRAME_SIZE] != RECV_FRAME_HEAD:
            return False
        return True

    def pop_frame(self) -> FeedbackFrame | None:
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
            return FeedbackFrame(
                motor_id=int(parsed[1]),
                state=int(parsed[2]),
                position=float(parsed[3]),
                velocity=float(parsed[4]),
                torque=float(parsed[5]),
                mos_temperature=float(parsed[6]),
            )


class MotorSequenceChecker:
    def __init__(self, motor_ids: tuple[int, ...]) -> None:
        if not motor_ids:
            raise ValueError("motor_ids must not be empty.")
        self._motor_ids = tuple(int(motor_id) for motor_id in motor_ids)
        self._expected_motor_id: int | None = None
        self.error_count = 0

    def reset(self) -> None:
        self._expected_motor_id = None

    def _next_motor_id(self, current_motor_id: int) -> int:
        index = self._motor_ids.index(int(current_motor_id))
        return self._motor_ids[(index + 1) % len(self._motor_ids)]

    def observe(self, motor_id: int) -> bool:
        motor_id = int(motor_id)
        if motor_id not in self._motor_ids:
            self.error_count += 1
            self._expected_motor_id = None
            return False
        if self._expected_motor_id is None:
            self._expected_motor_id = self._next_motor_id(motor_id)
            return True

        ok = motor_id == self._expected_motor_id
        if not ok:
            self.error_count += 1
        self._expected_motor_id = self._next_motor_id(motor_id)
        return ok


class SingleMotorCommandAdapter:
    def __init__(self, *, motor_count: int = 7) -> None:
        if int(motor_count) != 7:
            raise ValueError("The current UART adapter expects exactly 7 motor slots.")
        self._motor_count = int(motor_count)

    def pack(self, motor_id: int, command: float) -> bytes:
        motor_id = int(motor_id)
        if not 1 <= motor_id <= self._motor_count:
            raise ValueError(f"motor_id must be within [1, {self._motor_count}].")
        payload = np.zeros(self._motor_count, dtype=np.float32)
        payload[motor_id - 1] = np.float32(command)
        payload_bytes = COMMAND_PAYLOAD_STRUCT.pack(*[float(value) for value in payload])
        checksum = calculate_xor_checksum(COMMAND_FRAME_HEAD + payload_bytes)
        return COMMAND_FRAME_HEAD + payload_bytes + bytes((checksum,)) + COMMAND_FRAME_TAIL


__all__ = [
    "COMMAND_FRAME_HEAD",
    "COMMAND_FRAME_SIZE",
    "COMMAND_FRAME_TAIL",
    "COMMAND_PAYLOAD_STRUCT",
    "FeedbackFrame",
    "MotorSequenceChecker",
    "RECV_FRAME_HEAD",
    "RECV_FRAME_SIZE",
    "RECV_FRAME_STRUCT",
    "SerialFrameParser",
    "SingleMotorCommandAdapter",
    "calculate_xor_checksum",
]
