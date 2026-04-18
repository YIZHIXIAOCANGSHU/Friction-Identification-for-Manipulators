from __future__ import annotations

from typing import Protocol

from friction_identification_core.config import SerialConfig


class SerialTransport(Protocol):
    def read(self, size: int) -> bytes:
        ...

    def write(self, payload: bytes) -> int:
        ...

    def reset_input_buffer(self) -> None:
        ...

    def close(self) -> None:
        ...


class PySerialTransport:
    def __init__(self, config: SerialConfig) -> None:
        import serial

        self._serial = serial.Serial(
            port=config.port,
            baudrate=config.baudrate,
            timeout=float(config.read_timeout),
            write_timeout=float(config.write_timeout),
        )

    def read(self, size: int) -> bytes:
        return bytes(self._serial.read(max(int(size), 1)))

    def write(self, payload: bytes) -> int:
        return int(self._serial.write(payload))

    def reset_input_buffer(self) -> None:
        if hasattr(self._serial, "reset_input_buffer"):
            self._serial.reset_input_buffer()

    def close(self) -> None:
        self._serial.close()


def open_serial_transport(config: SerialConfig) -> SerialTransport:
    return PySerialTransport(config)
