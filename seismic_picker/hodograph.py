"""Travel-time curves from confirmed picks, with conservative shot geometry classification.

Never infer a reciprocal/overtaking relationship from line slopes alone.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HodoPoint:
    path: Path
    wave: str
    trace_index: int
    src_id: int
    src_x: float
    rcv_x: float
    time_ms: float


@dataclass(frozen=True)
class HodoSeries:
    path: Path
    wave: str
    src_id: int
    src_x: float
    points: tuple[HodoPoint,...]
    receiver_limits: tuple[float,float]


def from_workspace(library):
    result=[]
    for path,session in library.entries.items():
        if not session.geometry:continue
        layers=session.wave_layers or {'0':session.layer('0')}
        for code,layer in layers.items():
            pts=[]
            for i,sample in sorted(layer.picks.items()):
                if not (0 <= i < len(session.geometry)):continue
                h=session.geometry[i]
                dt=session.dt_us or 0
                if dt <= 0:continue
                pts.append(HodoPoint(path,code,int(i),int(h['src_id']),float(h['src_x']),
                                     float(h['rcv_x']),float(h['delay_ms'])+sample*dt/1000))
            if pts:
                pts.sort(key=lambda p:p.rcv_x)
                all_x=[float(h['rcv_x']) for h in session.geometry]
                result.append(HodoSeries(path,code,pts[0].src_id,pts[0].src_x,
                                         tuple(pts),(min(all_x),max(all_x))))
    return result


def geometry_relation(a:HodoSeries,b:HodoSeries) -> str:
    """Classify sources relative to overlapping receiver spread; ambiguous -> unknown.

    For shared receiver coverage, opposite outside sources give reciprocal shots;
    same-side outside sources give overtaking shots. Inside/degenerate geometry is
    deliberately left unclassified and must be interpreted by a geophysicist.
    """
    if a.path==b.path:return 'один ПВ'
    left=max(a.receiver_limits[0],b.receiver_limits[0])
    right=min(a.receiver_limits[1],b.receiver_limits[1])
    if left>=right:return 'нет общего участка'
    def side(x):
        if x<=left+1e-8:return -1
        if x>=right-1e-8:return 1
        return 0
    sa,sb=side(a.src_x),side(b.src_x)
    if not sa or not sb:return 'неоднозначная геометрия'
    if sa!=sb:return 'встречные'
    return 'нагоняющие'
