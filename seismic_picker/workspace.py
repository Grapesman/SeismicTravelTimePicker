"""Per-file picking sessions, loaded on demand instead of keeping hundreds of SEG-Y arrays.

A workspace preserves verified picks, unapproved proposals, viewport and per-file display
settings across switching and optionally across application restarts. Original SEG-Y data are
always read from disk; the workspace never rewrites source recordings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json

from .tracking import ProposalDetail
from .waves import WaveLayer, WaveType


@dataclass
class ShotSession:
    path: Path
    picks: dict[int, int] = field(default_factory=dict)
    proposals: dict[int, int] = field(default_factory=dict)
    proposal_info: dict[int, ProposalDetail] = field(default_factory=dict)
    failed_intervals: tuple[tuple[int, int], ...] = ()
    history: list[dict[int, int]] = field(default_factory=list)
    selected_trace: int | None = None
    dirty: bool = False
    view: tuple[tuple[float, float], tuple[float, float]] | None = None
    gain: float = 1.1
    agc: bool = True
    color: bool = True
    n_traces: int | None = None
    n_samples: int | None = None
    dt_us: int | None = None
    file_size: int | None = None
    wave_layers: dict[str,WaveLayer] = field(default_factory=dict)
    active_wave: str = '0'
    geometry: list[dict] = field(default_factory=list)

    def layer(self,code: str) -> WaveLayer:
        if code not in self.wave_layers:
            if code == '0' and self.picks:
                self.wave_layers[code]=WaveLayer(self.picks.copy(),self.proposals.copy(),
                                                  self.proposal_info.copy(),self.failed_intervals,
                                                  [h.copy() for h in self.history],self.selected_trace)
            else:
                self.wave_layers[code]=WaveLayer()
        return self.wave_layers[code]

    def validate_record(self, record):
        if self.n_traces is not None and (self.n_traces, self.n_samples, self.dt_us) != (
                record.n_traces, record.n_samples, record.dt_us):
            raise ValueError('SEG-Y изменился: число трасс, отсчётов или шаг времени не совпадает с сохранённым сеансом.')
        if self.file_size is not None and record.path.stat().st_size != self.file_size:
            raise ValueError('Размер SEG-Y изменился после сохранения сеанса.')
        for layer in [self.layer('0'), *[layer for code,layer in self.wave_layers.items() if code!='0']]:
          for trace, sample in {**layer.picks, **layer.proposals}.items():
            if not (0 <= trace < record.n_traces and 0 <= sample < record.n_samples):
                raise ValueError('Точки не соответствуют размерам текущего SEG-Y.')

    def to_dict(self, parent: Path) -> dict:
        try:
            relative_path = str(self.path.relative_to(parent))
        except ValueError:
            relative_path = None
        return {
            'path': str(self.path), 'relative_path': relative_path,
            'picks': {str(k): v for k,v in self.picks.items()},
            'proposals': {str(k): v for k,v in self.proposals.items()},
            'proposal_info': {str(k): {'sample': p.sample, 'confidence': p.confidence,
                                     'similarity': p.similarity, 'deviation_ms': p.deviation_ms,
                                     'quality': p.quality} for k,p in self.proposal_info.items()},
            'failed_intervals': [list(pair) for pair in self.failed_intervals],
            'selected_trace': self.selected_trace, 'view': self.view,
            'gain': self.gain, 'agc': self.agc, 'color': self.color,
            'n_traces': self.n_traces, 'n_samples': self.n_samples,
            'dt_us': self.dt_us, 'file_size': self.file_size,
            'wave_layers':{code:layer.to_dict() for code,layer in self.wave_layers.items()},
            'active_wave':self.active_wave, 'geometry':self.geometry,
        }

    @classmethod
    def from_dict(cls, data: dict, parent: Path) -> 'ShotSession':
        p=Path(data['path']).expanduser()
        if not p.is_file() and data.get('relative_path'):
            alt=parent / data['relative_path']
            if alt.is_file():p=alt
        session=cls(path=p.resolve())
        session.picks={int(k):int(v) for k,v in data.get('picks',{}).items()}
        session.proposals={int(k):int(v) for k,v in data.get('proposals',{}).items()}
        session.proposal_info={int(k):ProposalDetail(**v) for k,v in data.get('proposal_info',{}).items()}
        session.failed_intervals=tuple(tuple(map(int,pair)) for pair in data.get('failed_intervals',()))
        session.selected_trace=data.get('selected_trace')
        session.wave_layers={str(k):WaveLayer.from_dict(v) for k,v in data.get('wave_layers',{}).items()}
        session.active_wave=str(data.get('active_wave','0'))
        session.geometry=list(data.get('geometry',[]))
        if data.get('view'):
            session.view=(tuple(map(float,data['view'][0])), tuple(map(float,data['view'][1])))
        session.gain=float(data.get('gain',1.1))
        session.agc=bool(data.get('agc',True))
        session.color=bool(data.get('color',True))
        for key in ('n_traces','n_samples','dt_us','file_size'):
            setattr(session,key,data.get(key))
        return session


class SessionLibrary:
    """Insertion-ordered set of SEG-Y sessions; opening 200 filenames does not load 200 arrays."""
    def __init__(self):
        self.entries: dict[Path,ShotSession]={}
        self.wave_types: dict[str,WaveType]={'0':WaveType()}

    def add(self, files) -> list[Path]:
        added=[]
        for name in files:
            path=Path(name).expanduser().resolve()
            if path not in self.entries:
                self.entries[path]=ShotSession(path)
                added.append(path)
        return added

    def save(self, path: str | Path, current: Path | None) -> None:
        target=Path(path).resolve()
        data={'schema_version':3, 'current':str(current) if current else None,
              'wave_types':{code:w.to_dict() for code,w in self.wave_types.items()},
              'sessions':[s.to_dict(target.parent) for s in self.entries.values()]}
        # Atomic write so an interrupted save does not corrupt the only session copy.
        temp=target.with_name(target.name+'.tmp')
        try:
            temp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
            temp.replace(target)
        finally:
            if temp.exists():temp.unlink()

    @classmethod
    def load(cls, path: str | Path):
        source=Path(path).resolve()
        data=json.loads(source.read_text(encoding='utf-8'))
        if data.get('schema_version') not in (2,3):
            raise ValueError('Неизвестный формат многосейсмограммного сеанса.')
        library=cls()
        library.wave_types={str(k):WaveType.from_dict(v) for k,v in data.get('wave_types', {'0':WaveType().to_dict()}).items()}
        library.wave_types.setdefault('0',WaveType())
        for item in data['sessions']:
            entry=ShotSession.from_dict(item,source.parent)
            if entry.path in library.entries:
                raise ValueError('Дублирование файла в сеансе.')
            library.entries[entry.path]=entry
        current=Path(data['current']).resolve() if data.get('current') else None
        if current not in library.entries:
            current=next(iter(library.entries),None)
        return library,current
