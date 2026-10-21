"""Conservative phase-consistent tracking between *manual* SEG-Y anchors.

No black-box training, no alteration of original trace amplitudes, and no
automatic acceptance. Dynamic programming (Viterbi) chooses a globally coherent
sequence of extrema in a user-defined time corridor. The confidence score is a
heuristic review priority, not a calibrated probability or geophysical proof.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Mapping
import numpy as np

from .picking import extrema
from .segy import SeismicRecord


@dataclass(frozen=True)
class ProposalDetail:
    sample: int
    confidence: float
    similarity: float
    deviation_ms: float
    quality: str  # 'высокая', 'средняя', 'проверить'


@dataclass(frozen=True)
class TrackResult:
    points: dict[int, ProposalDetail]
    failed_intervals: tuple[tuple[int, int], ...]


def _profile(signal: np.ndarray, sample: int, half_width: int) -> np.ndarray:
    """Wavelet window relative to picked extremum, padded with zeros."""
    dst = np.zeros(2 * half_width + 1, dtype=np.float64)
    left = max(0, sample - half_width)
    right = min(len(signal), sample + half_width + 1)
    a = half_width - (sample - left)
    dst[a:a + (right - left)] = signal[left:right]
    dst -= dst.mean()
    norm = np.linalg.norm(dst)
    return dst / norm if norm > 1e-12 else dst


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def _trace_dt(record: SeismicRecord, trace: int, sample: int) -> float:
    return record.time_ms(trace, sample)


def follow_phase(record: SeismicRecord, anchors: Mapping[int, int],
                 radius_ms: float = 6.0, polarity: str = 'positive',
                 signals: np.ndarray | None = None, max_candidates: int = 14,
                 min_confidence: float = 0.0) -> TrackResult:
    """Suggest trace-by-trace matching extrema between confirmed anchors.

    - Tracking never extends *outside* adjacent anchors.
    - Every trace in an interval must have a candidate (otherwise interval
      stays unproposed rather than inventing intermediate samples).
    - All time math uses raw SEG-Y sample indices, microsecond interval and
      per-trace delay. Display gain does not affect the exported time.
    - Only *suggestions* returned; the GUI explicitly asks before acceptance.
    """
    if not (0.25 <= radius_ms <= 100):
        raise ValueError('Коридор поиска должен составлять 0,25–100 мс.')
    if polarity not in ('positive', 'negative', 'both'):
        raise ValueError('Неизвестная полярность пика.')
    if not (2 <= max_candidates <= 60):
        raise ValueError('Количество кандидатов вне допустимого диапазона.')
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError('Минимальная уверенность должна быть от 0 до 1.')
    signal = np.asarray(record.amplitudes if signals is None else signals)
    if signal.shape != record.amplitudes.shape:
        raise ValueError('Размер массива амплитуд не соответствует сейсмограмме.')
    sorted_anchors = sorted((int(k), int(v)) for k, v in anchors.items())
    for k, v in sorted_anchors:
        if not (0 <= k < record.n_traces and 0 <= v < record.n_samples):
            raise ValueError('Опорная точка за пределами сейсмограммы.')
    if len(sorted_anchors) < 2:
        return TrackResult({}, ())

    # Profiles are short relative to full traces (default approx 5 ms total).
    half_width = max(3, int(round(2.5 * 1000 / record.dt_us)))
    result: dict[int, ProposalDetail] = {}
    failed: list[tuple[int, int]] = []
    coords = record.receiver_x
    for (left, l_sample), (right, r_sample) in zip(sorted_anchors, sorted_anchors[1:]):
        if right - left <= 1:
            continue
        # Use receiver positions for physical interpolation, fall back to
        # sequential order if positions are duplicated / nonmonotonic.
        xl, xr = float(coords[left]), float(coords[right])
        use_coords = (abs(xr - xl) > 1e-10 and
                      np.all(np.diff(coords[left:right + 1]) * (xr - xl) > 0))
        tl, tr = _trace_dt(record, left, l_sample), _trace_dt(record, right, r_sample)
        n = right - left
        traces = list(range(left + 1, right))
        candidate_samples: list[np.ndarray] = [np.array([l_sample], dtype=np.int32)]
        candidate_times: list[np.ndarray] = [np.array([tl], dtype=float)]
        expected = [tl]
        ok = True
        for trace in traces:
            fraction = ((float(coords[trace]) - xl) / (xr - xl)
                        if use_coords else (trace - left) / n)
            tt = tl + fraction * (tr - tl)
            expected.append(tt)
            start = int(np.ceil((tt - radius_ms - record.headers[trace].delay_ms) * 1000 / record.dt_us))
            end = int(np.floor((tt + radius_ms - record.headers[trace].delay_ms) * 1000 / record.dt_us))
            candidates = extrema(signal[trace], start, end, polarity)
            if len(candidates) == 0:
                ok = False
                break
            dt_candidates = np.array([_trace_dt(record, trace, int(s)) for s in candidates])
            if len(candidates) > max_candidates:
                # Keep strongest candidates near interpolated trajectory;
                # avoid combinatorial blowups for noisy high-frequency traces.
                strength = np.abs(signal[trace, candidates])
                strength /= max(np.quantile(strength, .9), 1e-12)
                priority = (np.abs(dt_candidates - tt) / max(radius_ms, .25)
                            - .32 * np.clip(strength, 0, 3))
                inds = np.sort(np.argsort(priority)[:max_candidates])
                candidates = candidates[inds]
                dt_candidates = dt_candidates[inds]
            candidate_samples.append(candidates)
            candidate_times.append(dt_candidates)
        if not ok:
            failed.append((left, right))
            continue
        candidate_samples.append(np.array([r_sample], dtype=np.int32))
        candidate_times.append(np.array([tr], dtype=float))
        expected.append(tr)
        inds = [left] + traces + [right]
        profiles = [[_profile(signal[trace], int(s), half_width) for s in cand]
                    for trace, cand in zip(inds, candidate_samples)]
        # Min-cost path with three pieces: interpolate, wavelet similarity,
        # and a smooth local time derivative between successive receivers.
        scores = np.zeros(1, dtype=np.float64)
        paths: list[np.ndarray] = []
        smooth = max(.8, radius_ms / 2.5)
        for j in range(1, len(inds)):
            prev_t, curr_t = candidate_times[j - 1], candidate_times[j]
            base_slope = expected[j] - expected[j - 1]
            cost = np.empty((len(prev_t), len(curr_t)))
            for a, tprev in enumerate(prev_t):
                for b, tcur in enumerate(curr_t):
                    similarity = _correlation(profiles[j - 1][a], profiles[j][b])
                    phase_change = 0.0
                    va = float(signal[inds[j - 1], candidate_samples[j - 1][a]])
                    vb = float(signal[inds[j], candidate_samples[j][b]])
                    if va * vb < 0: phase_change = 3.5
                    cost[a, b] = (1.6 * (1 - similarity)
                                  + .7 * ((tcur - tprev - base_slope) / smooth) ** 2
                                  + phase_change)
            current = scores[:, None] + cost
            arg = np.argmin(current, axis=0)
            scores = current[arg, np.arange(len(curr_t))]
            if j < len(inds) - 1:
                delta = (curr_t - expected[j]) / radius_ms
                scores += .65 * (delta ** 2)
                # prefer a well-expressed peak, but do not reward exaggerated
                # amplitudes enough to override phase continuity.
                strengths = np.abs(signal[inds[j], candidate_samples[j]])
                relative = strengths / max(float(np.max(strengths)), 1e-12)
                scores += .18 * (1 - relative)
            paths.append(arg)
        pos = int(np.argmin(scores))
        selected = [pos]
        for j in range(len(paths) - 1, -1, -1):
            pos = int(paths[j][pos])
            selected.append(pos)
        selected.reverse()   # one selected index per anchor+intermediate trace
        for j in range(1, len(inds) - 1):
            trace = inds[j]
            c = selected[j]
            sample = int(candidate_samples[j][c])
            tt = _trace_dt(record, trace, sample)
            dev = abs(tt - expected[j])
            prev_sim = _correlation(profiles[j-1][selected[j-1]], profiles[j][c])
            next_sim = _correlation(profiles[j][c], profiles[j+1][selected[j+1]])
            similarity = min(prev_sim, next_sim)
            amp = abs(float(signal[trace, sample]))
            # Local data strength, using the local 10 ms neighbourhood.
            half = max(4, round(5 * 1000 / record.dt_us))
            segment = signal[trace, max(sample-half, 0): min(sample+half+1, record.n_samples)]
            rms = float(np.sqrt(np.mean(segment * segment))) if segment.size else 0.0
            strength_ratio = amp / (rms + 1e-12)
            strength_score = np.clip(strength_ratio/2, 0.0, 1.0)
            conf = float(np.clip(.50 * ((similarity + 1) / 2)
                                 + .30 * np.exp(-dev / max(radius_ms, .25))
                                 + .20 * strength_score, 0, 1))
            quality = 'высокая' if conf >= .78 else 'средняя' if conf >= .59 else 'проверить'
            if conf >= min_confidence:
                result[trace] = ProposalDetail(sample, round(conf, 3), round(similarity, 3), round(dev, 3), quality)
    return TrackResult(result, tuple(failed))
