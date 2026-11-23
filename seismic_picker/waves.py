"""Typed, multi-wave picking state. One receiver may carry multiple wave arrivals.

The GUI owns one active picker at a time; inactive layers remain independent.
Only *confirmed* picks are used for output/hodographs.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import re
from .tracking import ProposalDetail

_COLORS = ('#10a56a', '#d97730', '#237bb6', '#9646b2', '#ba5370', '#687b24')


def valid_code(code: str) -> str:
    code = str(code).strip()
    if not re.fullmatch(r'[\w.\-]{1,32}', code, flags=re.UNICODE):
        raise ValueError('Код волны: от 1 до 32 букв, цифр, точек, дефисов или подчёркиваний; без пробелов.')
    return code


def wave_color(index: int) -> str:
    return _COLORS[index % len(_COLORS)]


@dataclass
class WaveType:
    code: str = '0'
    name: str = 'Головная волна'
    line_color: str = '#00a65c'
    peak_color: str = '#ffda1f'
    guide_color: str = '#f59e0b'

    def __post_init__(self):
        self.code = valid_code(self.code)
        if not self.name.strip():
            raise ValueError('Название волны не может быть пустым.')
        for c in (self.line_color, self.peak_color, self.guide_color):
            if not re.fullmatch(r'#[0-9a-fA-F]{6}', c):
                raise ValueError('Цвет должен иметь формат #RRGGBB.')

    def to_dict(self) -> dict:
        return vars(self).copy()

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


@dataclass
class WaveLayer:
    picks: dict[int,int] = field(default_factory=dict)
    proposals: dict[int,int] = field(default_factory=dict)
    proposal_info: dict[int,ProposalDetail] = field(default_factory=dict)
    failed_intervals: tuple[tuple[int,int],...] = ()
    history: list[dict[int,int]] = field(default_factory=list)
    selected_trace: int | None = None

    def to_dict(self) -> dict:
        return {
            'picks': {str(k):int(v) for k,v in self.picks.items()},
            'proposals': {str(k):int(v) for k,v in self.proposals.items()},
            'proposal_info': {str(k):vars(p).copy() for k,p in self.proposal_info.items()},
            'failed_intervals': [list(p) for p in self.failed_intervals],
            'history': [{str(k):v for k,v in snap.items()} for snap in self.history],
            'selected_trace':self.selected_trace,
        }

    @classmethod
    def from_dict(cls, data:dict):
        return cls(
            picks={int(k):int(v) for k,v in data.get('picks',{}).items()},
            proposals={int(k):int(v) for k,v in data.get('proposals',{}).items()},
            proposal_info={int(k):ProposalDetail(**p) for k,p in data.get('proposal_info',{}).items()},
            failed_intervals=tuple(tuple(map(int,p)) for p in data.get('failed_intervals',())),
            history=[{int(k):int(v) for k,v in snap.items()} for snap in data.get('history',[])],
            selected_trace=data.get('selected_trace'),
        )


def rows_for_layer(record, layer: WaveLayer, code: str, all_traces: bool=False):
    """Use the trace *headers* as the authoritative output metadata."""
    valid_code(code)
    rows=[]
    for i,h in enumerate(record.headers):
        if i not in layer.picks and not all_traces:
            continue
        t=record.time_ms(i,layer.picks[i]) if i in layer.picks else None
        rows.append((h.src_id, h.src_x, 0, h.rcv_id, h.rcv_x, 0, t, code))
    return rows
