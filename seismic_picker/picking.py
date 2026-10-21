"""Deterministic sample-index picking, interpolation proposals and exports."""
from __future__ import annotations
from dataclasses import dataclass, field
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from .segy import SeismicRecord


def extrema(signal: np.ndarray, low: int, high: int, polarity: str = 'positive') -> np.ndarray:
    """Return indices of local positive peaks / negative troughs in inclusive interval."""
    low = max(1, int(low)); high = min(len(signal)-2, int(high))
    if low > high:
        return np.array([], dtype=np.int32)
    i = np.arange(low, high+1)
    previous, current, following = signal[i-1], signal[i], signal[i+1]
    pos = (current > 0) & (current >= previous) & (current > following)
    neg = (current < 0) & (current <= previous) & (current < following)
    if polarity == 'positive': cond = pos
    elif polarity == 'negative': cond = neg
    else: cond = pos | neg
    return i[cond]


def nearest_peak(signal: np.ndarray, target_sample: float,
                 radius_samples: int, polarity: str = 'positive') -> int | None:
    peaks = extrema(signal, round(target_sample)-radius_samples,
                    round(target_sample)+radius_samples, polarity)
    if not len(peaks):
        return None
    # First priority is time continuity; peak amplitude only breaks near-ties.
    best = np.argmin(np.abs(peaks-target_sample))
    return int(peaks[best])


@dataclass
class PickerState:
    record: SeismicRecord
    picks: dict[int, int] = field(default_factory=dict)
    proposals: dict[int, int] = field(default_factory=dict)
    proposal_info: dict[int, object] = field(default_factory=dict)
    failed_intervals: tuple[tuple[int, int], ...] = ()

    def set_pick(self, trace: int, sample: int) -> None:
        if not (0 <= trace < self.record.n_traces and 0 <= sample < self.record.n_samples):
            raise ValueError('Точка выходит за пределы сейсмограммы.')
        # A change of an anchor invalidates every derived suggestion.
        # Never leave a stale computed path in the user's view.
        if self.proposals:
            self.clear_proposals()
        self.picks[int(trace)] = int(sample)

    def remove(self, trace: int) -> None:
        if trace in self.picks and self.proposals:
            self.clear_proposals()
        self.picks.pop(trace, None)
        self.proposals.pop(trace, None)
        self.proposal_info.pop(trace, None)

    def suggest_between(self, radius_ms: float = 6.0, polarity: str = 'positive', signals: np.ndarray | None = None) -> int:
        """Interpolate time between manual anchors, snap to local extrema as *unapproved* proposals."""
        self.clear_proposals()
        anchors = sorted(self.picks)
        radius = max(1, int(round(radius_ms * 1000/self.record.dt_us)))
        for left, right in zip(anchors, anchors[1:]):
            if right - left <= 1:
                continue
            t_left = self.record.time_ms(left, self.picks[left])
            t_right = self.record.time_ms(right, self.picks[right])
            x_left = self.record.headers[left].rcv_x
            x_right = self.record.headers[right].rcv_x
            for trace in range(left+1, right):
                if trace in self.picks:
                    continue
                x = self.record.headers[trace].rcv_x
                fraction = (x-x_left)/(x_right-x_left) if x_right != x_left else (trace-left)/(right-left)
                t = t_left + (t_right-t_left)*fraction
                estimated_idx = (t-self.record.headers[trace].delay_ms)*1000/self.record.dt_us
                peak = nearest_peak(signals[trace] if signals is not None else self.record.amplitudes[trace],
                                    estimated_idx, radius, polarity)
                if peak is not None:
                    self.proposals[trace] = peak
        return len(self.proposals)

    def suggest_phase(self, radius_ms: float = 6.0, polarity: str = 'positive',
                      signals: np.ndarray | None = None) -> int:
        """Dynamic-programming phase tracker, never accepts candidates automatically."""
        from .tracking import follow_phase
        result = follow_phase(self.record, self.picks, radius_ms, polarity, signals=signals)
        self.clear_proposals()
        self.proposals = {trace: detail.sample for trace, detail in result.points.items()}
        self.proposal_info = result.points.copy()
        self.failed_intervals = result.failed_intervals
        return len(self.proposals)

    def clear_proposals(self) -> None:
        self.proposals.clear()
        self.proposal_info.clear()
        self.failed_intervals = ()

    def accept(self) -> int:
        count = len(self.proposals)
        self.picks.update(self.proposals)
        self.clear_proposals()
        return count

    def rows(self, all_traces: bool = False) -> list[tuple]:
        result = []
        for i,h in enumerate(self.record.headers):
            if i not in self.picks and not all_traces:
                continue
            tt = self.record.time_ms(i, self.picks[i]) if i in self.picks else None
            result.append((h.src_id, h.src_x, 0, h.rcv_id, h.rcv_x, 0, tt, 0))
        return result

    def save_project(self, path: str | Path) -> None:
        data = {
            'schema_version': 1,
            'source_path': str(self.record.path),
            'source_sha256': sha256(self.record.path),
            'sample_interval_us': self.record.dt_us,
            'trace_count': self.record.n_traces,
            'sample_count': self.record.n_samples,
            'picks': [{'trace_index': i, 'sample_index': s} for i,s in sorted(self.picks.items())],
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

    def load_project(self, path: str | Path) -> None:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        if data.get('schema_version') != 1:
            raise ValueError('Неизвестная версия проекта.')
        if data.get('source_sha256') != sha256(self.record.path):
            raise ValueError('SGY-файл не совпадает с файлом проекта (SHA-256).')
        if data['trace_count'] != self.record.n_traces or data['sample_count'] != self.record.n_samples:
            raise ValueError('Размеры сейсмограммы не совпадают с проектом.')
        next_picks = {}
        for row in data['picks']:
            i, s = int(row['trace_index']), int(row['sample_index'])
            if not (0 <= i < self.record.n_traces and 0 <= s < self.record.n_samples):
                raise ValueError('В проекте обнаружены некорректные точки.')
            next_picks[i] = s
        self.picks = next_picks
        self.clear_proposals()

    def export_txt(self, path: str | Path, all_traces: bool = False) -> None:
        export_table(path, self.rows(all_traces=all_traces), delimiter='\t')

    def export_csv(self, path: str | Path, all_traces: bool = False) -> None:
        export_table(path, self.rows(all_traces=all_traces), delimiter=';')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def _format(v) -> str:
    if v is None:
        return ''
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return f'{float(v):.6f}'.rstrip('0').rstrip('.') or '0'


def export_table(path: str | Path, rows: list[tuple], delimiter: str = '\t') -> None:
    path = Path(path)
    with path.open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file, delimiter=delimiter)
        writer.writerow(['SRC_ID', 'SRC_X', 'SRC_Z', 'RCV_ID', 'RCV_X', 'RCV_Z', 'Traveltime', 'Wave'])
        writer.writerows([_format(v) for v in row] for row in rows)
