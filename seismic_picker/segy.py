"""Strict, read-only SEG-Y loader for shot-gather travel-time picking.

SEG-Y bytes/positions below use the conventional one-based specification.
Coordinates are scaled according to trace header bytes 71-72.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
import numpy as np

FORMATS = {1: (4, "IBM 32-bit float"), 2: (4, "int32"), 3: (2, "int16"),
           5: (4, "IEEE float32"), 6: (8, "IEEE float64"), 8: (1, "int8")}


class SegyError(ValueError):
    """User-facing unsupported or damaged SEG-Y file."""


@dataclass(frozen=True)
class TraceHeader:
    seq_line: int
    ffid: int
    sp: int
    src_x: float
    rcv_x: float
    dt_us: int
    delay_ms: float
    coord_scalar: int

    @property
    def src_id(self) -> int:
        return self.ffid or self.sp

    @property
    def rcv_id(self) -> int:
        return self.seq_line


@dataclass
class SeismicRecord:
    path: Path
    headers: list[TraceHeader]
    amplitudes: np.ndarray    # [trace, sample] float64
    dt_us: int
    endian: str
    format_code: int

    @property
    def n_traces(self) -> int:
        return len(self.headers)

    @property
    def n_samples(self) -> int:
        return int(self.amplitudes.shape[1])

    @property
    def receiver_x(self) -> np.ndarray:
        return np.array([h.rcv_x for h in self.headers], dtype=np.float64)

    def time_ms(self, trace_idx: int, sample_idx: float) -> float:
        h = self.headers[trace_idx]
        return h.delay_ms + float(sample_idx) * self.dt_us / 1000.0

    def times_ms(self, trace_idx: int) -> np.ndarray:
        return self.headers[trace_idx].delay_ms + np.arange(self.n_samples) * (self.dt_us / 1000.0)


def _u16(buf: bytes, offset: int, e: str) -> int:
    return struct.unpack_from(e + 'H', buf, offset)[0]


def _i16(buf: bytes, offset: int, e: str) -> int:
    return struct.unpack_from(e + 'h', buf, offset)[0]


def _i32(buf: bytes, offset: int, e: str) -> int:
    return struct.unpack_from(e + 'i', buf, offset)[0]


def _scale_coord(value: int, scalar: int) -> float:
    if scalar == 0:
        return float(value)
    return float(value * scalar) if scalar > 0 else float(value / abs(scalar))


def _ibm_float32(raw: bytes, endian: str) -> np.ndarray:
    bits = np.frombuffer(raw, dtype=endian+'u4').astype(np.uint32)
    fraction = (bits & 0x00FFFFFF).astype(np.float64) / (2.0 ** 24)
    exponent = ((bits >> 24) & 0x7F).astype(np.float64) - 64
    sign = np.where((bits & 0x80000000) != 0, -1., 1.)
    return sign * fraction * np.power(16.0, exponent)


def _decode_samples(raw: bytes, code: int, endian: str) -> np.ndarray:
    if code == 1:
        return _ibm_float32(raw, endian)
    types = {2: 'i4', 3: 'i2', 5: 'f4', 6: 'f8', 8: 'i1'}
    return np.frombuffer(raw, dtype=endian+types[code]).astype(np.float64)


def _choose_endian(binary: bytes, file_size: int, forced: str | None) -> tuple[str, int, int, int, int]:
    candidates = []
    for endian in ('<', '>') if forced is None else (forced,):
        dt = _u16(binary, 16, endian)
        ns = _u16(binary, 20, endian)
        fmt = _u16(binary, 24, endian)
        if fmt not in FORMATS or not (1 <= ns <= 65535) or not (1 <= dt <= 65535):
            continue
        # SEG-Y Rev1: binary header bytes 3505-3506 => zero-based offset 304
        extended_headers = _i16(binary, 304, endian)
        if extended_headers < 0 or extended_headers > 10000:
            continue
        start = 3600 + extended_headers * 3200
        trace_bytes = 240 + ns * FORMATS[fmt][0]
        fits = file_size >= start + trace_bytes and (file_size-start) % trace_bytes == 0
        candidates.append((fits, endian, dt, ns, fmt, start))
    if not candidates:
        raise SegyError('Невозможно определить формат SEG-Y. Проверьте бинарный заголовок и порядок байтов.')
    candidates.sort(key=lambda c: c[0], reverse=True)
    fits, endian, dt, ns, fmt, start = candidates[0]
    if not fits:
        raise SegyError('Размер файла не соответствует регулярным трассам SEG-Y; переменная длина пока не поддерживается.')
    return endian, dt, ns, fmt, start


def read_segy(path: str | Path, endian: str | None = None) -> SeismicRecord:
    """Read fixed-length, uncompressed SEG-Y (Rev0/1), little or big endian.

    Does not execute data, modify the original file or guess undocumented trace headers.
    """
    path = Path(path)
    if endian not in (None, '<', '>'):
        raise ValueError("endian must be None, '<' or '>'")
    file_size = path.stat().st_size
    if file_size < 3844:
        raise SegyError('Файл слишком маленький для SEG-Y.')
    with path.open('rb') as f:
        f.seek(3200)
        binary = f.read(400)
        e, dt, ns, fmt, trace_offset = _choose_endian(binary, file_size, endian)
        bytes_per_sample = FORMATS[fmt][0]
        trace_size = 240 + ns * bytes_per_sample
        n_traces = (file_size-trace_offset)//trace_size
        # A practical memory ceiling; larger files need lazy paging in a later release.
        if n_traces * ns > 45_000_000:
            raise SegyError('Слишком большой набор трасс для этой версии (лимит 45 млн отсчётов).')
        f.seek(trace_offset)
        headers: list[TraceHeader] = []
        samples = np.empty((n_traces, ns), dtype=np.float64)
        for i in range(n_traces):
            h = f.read(240)
            if len(h) != 240:
                raise SegyError(f'Оборван заголовок трассы {i+1}.')
            this_ns = _u16(h, 114, e) or ns
            this_dt = _u16(h, 116, e) or dt
            if this_ns != ns or this_dt != dt:
                raise SegyError(f'Трасса {i+1}: переменная длина или дискретизация не поддерживается.')
            cs = _i16(h, 70, e)
            headers.append(TraceHeader(
                seq_line=_i32(h, 0, e) or (i+1),
                ffid=_i32(h, 8, e), sp=_i32(h, 16, e),
                src_x=_scale_coord(_i32(h, 72, e), cs),
                rcv_x=_scale_coord(_i32(h, 80, e), cs),
                dt_us=this_dt,
                delay_ms=float(_i16(h, 108, e)),
                coord_scalar=cs))
            chunk = f.read(ns*bytes_per_sample)
            if len(chunk) != ns*bytes_per_sample:
                raise SegyError(f'Оборваны отсчёты трассы {i+1}.')
            samples[i] = _decode_samples(chunk, fmt, e)
        if not np.all(np.isfinite(samples)):
            raise SegyError('В данных присутствуют NaN/Infinity; требуется предварительная проверка SEG-Y.')
    return SeismicRecord(path.resolve(), headers, samples, dt, e, fmt)
