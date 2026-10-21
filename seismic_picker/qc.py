"""Conservative, advisory QC for manually selected seismic arrivals.

The check is intentionally heuristic: it never rejects a pick by itself,
and never changes original SEG-Y samples or exported traveltimes.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping
from .segy import SeismicRecord


@dataclass(frozen=True)
class PhaseConcern:
    trace: int
    sample: int
    message: str
    expected_ms: float | None = None
    deviation_ms: float | None = None


def check_phase_continuity(record: SeismicRecord, picks: Mapping[int, int], trace: int,
                           sample: int, tolerance_ms: float = 8.0,
                           check_polarity: bool = True) -> list[PhaseConcern]:
    """Flag unusual jumps compared to neighboring manually confirmed picks.

    Times are evaluated in ms. Coordinates are used for interpolation, and
    trace indices are used as a fallback where receiver coordinates repeat.
    A low-information situation is *not* declared normal/abnormal.
    """
    if not 0 <= trace < record.n_traces or not 0 <= sample < record.n_samples:
        raise ValueError('Точка вне границ записи')
    if tolerance_ms <= 0:
        raise ValueError('Порог должен быть положительным')
    others = sorted(i for i in picks if i != trace)
    if not others:
        return []
    x = record.receiver_x
    time = record.time_ms(trace, sample)
    left = [i for i in others if i < trace]
    right = [i for i in others if i > trace]
    near = []
    if left:
        near.append(left[-1])
    if right:
        near.append(right[0])

    def interp(a: int, b: int) -> float:
        xa, xb, xt = float(x[a]), float(x[b]), float(x[trace])
        if abs(xb-xa) < 1e-9:
            xa, xb, xt = float(a), float(b), float(trace)
        ta = record.time_ms(a, picks[a]); tb = record.time_ms(b, picks[b])
        return ta + (tb-ta)*(xt-xa)/(xb-xa)

    estimate = None
    # Prefer bracketing anchors; otherwise extrapolate from two recent anchors.
    if left and right:
        a, b = left[-1], right[0]
        if b-a <= 6:
            estimate = interp(a, b)
    elif len(left) >= 2 and trace-left[-1] <= 3 and left[-1]-left[-2] <= 3:
        estimate = interp(left[-2], left[-1])
    elif len(right) >= 2 and right[0]-trace <= 3 and right[1]-right[0] <= 3:
        estimate = interp(right[0], right[1])
    concerns = []
    if estimate is not None:
        delta = abs(time-estimate)
        if delta > tolerance_ms:
            concerns.append(PhaseConcern(
                trace, sample, f'Время отличается от продолжения линии соседних пиков на {delta:.2f} мс '
                f'(ожидалось примерно {estimate:.2f} мс, выбрано {time:.2f} мс).',
                expected_ms=estimate, deviation_ms=delta))
    # Single-anchor detection: large *per receiver interval* jump, not blanket
    # mismatch across large gaps in receiver numbering.
    if estimate is None and near:
        step = min(near, key=lambda i: abs(i-trace))
        if abs(step-trace) == 1:
            delta = abs(time-record.time_ms(step, picks[step]))
            if delta > tolerance_ms*1.5:
                concerns.append(PhaseConcern(
                    trace, sample,
                    f'Между соседними приёмниками скачок времени {delta:.2f} мс. '
                    'Это может быть переход на другую фазу.'))
    if check_polarity and near:
        amplitude = record.amplitudes[trace, sample]
        if amplitude != 0:
            neighbor = min(near, key=lambda i: abs(i-trace))
            neighbor_amplitude = record.amplitudes[neighbor, picks[neighbor]]
            if neighbor_amplitude != 0 and ((amplitude > 0) != (neighbor_amplitude > 0)):
                concerns.append(PhaseConcern(
                    trace, sample,
                    f'Полярность отличается от ближайшего подтверждённого пика '
                    f'(приёмник {record.headers[neighbor].rcv_id}).'))
    return concerns


def audit_picks(record: SeismicRecord, picks: Mapping[int, int],
                tolerance_ms: float = 8.0) -> dict[int, list[PhaseConcern]]:
    """Check chronological trace progression; advisory only, no changes."""
    checked: dict[int, int] = {}
    warnings: dict[int, list[PhaseConcern]] = {}
    for trace in sorted(picks):
        findings = check_phase_continuity(record, checked, trace, picks[trace], tolerance_ms)
        if findings:
            warnings[trace] = findings
        checked[trace] = picks[trace]
    return warnings
