"""Strict round-trip TXT/CSV import with SEG-Y geometry and sample-grid validation.

Nothing is written to a workspace until all rows in the import have been validated.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import csv
import math
from .waves import valid_code

FIELDS=('SRC_ID','SRC_X','SRC_Z','RCV_ID','RCV_X','RCV_Z','TRAVELTIME','WAVE')


@dataclass(frozen=True)
class ImportedPick:
    src_id: int
    src_x: float
    src_z: float
    rcv_id: int
    rcv_x: float
    rcv_z: float
    t_ms: float | None
    wave: str
    line: int


def _number(val, integer=False):
    f=float(str(val).strip().replace(',','.'))
    if not math.isfinite(f):raise ValueError('Число не является конечным.')
    if integer and not f.is_integer():raise ValueError('Ожидается целое число.')
    return int(f) if integer else f


def read_pick_file(file: str | Path) -> list[ImportedPick]:
    path=Path(file)
    content=path.read_text(encoding='utf-8-sig')
    text_lines=[(i,line) for i,line in enumerate(content.splitlines(),1) if line.strip() and not line.lstrip().startswith('#')]
    if not text_lines:raise ValueError('TXT не содержит данных.')
    first=text_lines[0][1]
    delimiter=next((d for d in ('\t',';',',') if d in first),None)
    def split(line):
        return next(csv.reader([line],delimiter=delimiter)) if delimiter else line.split()
    columns=[s.strip().upper() for s in split(first)]
    if len(columns)!=len(set(columns)) or set(FIELDS)-set(columns):
        raise ValueError('Нужны восемь столбцов: '+', '.join(FIELDS)+'.')
    rows=[]
    for lineno,line in text_lines[1:]:
        cells=split(line)
        if len(cells)!=len(columns):
            raise ValueError(f'Строка {lineno}: неверное количество столбцов.')
        d=dict(zip(columns,cells))
        try:
            t=None if not d['TRAVELTIME'].strip() else _number(d['TRAVELTIME'])
            if t is not None and t<0:raise ValueError('Traveltime не может быть отрицательным.')
            rows.append(ImportedPick(
                _number(d['SRC_ID'],True),_number(d['SRC_X']),_number(d['SRC_Z']),
                _number(d['RCV_ID'],True),_number(d['RCV_X']),_number(d['RCV_Z']),
                t,valid_code(d['WAVE'] or '0'),lineno))
        except ValueError as exc:
            raise ValueError(f'Строка {lineno}: {exc}') from exc
    if not rows:raise ValueError('В файле отсутствуют строки с результатами.')
    return rows


def validate_pick_import(rows: list[ImportedPick], records, coord_tolerance=0.001):
    """Match rows to unique input records and trace headers; reject missing and ambiguities.

    `records` may be full SeismicRecord or lightweight header-only SEG-Y descriptors.
    Return {Path: {wave_code: {trace_index: sample_index}}} on total success.
    """
    if not records:raise ValueError('Не выбраны SEG-Y для сопоставления.')
    result=defaultdict(lambda:defaultdict(dict))
    problems=[]
    for row in rows:
        matches=[]
        for rec in records:
            for idx,h in enumerate(rec.headers):
                if (h.src_id==row.src_id and abs(h.src_x-row.src_x)<=coord_tolerance and
                    h.rcv_id==row.rcv_id and abs(h.rcv_x-row.rcv_x)<=coord_tolerance):
                    matches.append((rec,idx))
        if len(matches)!=1:
            name='; '.join(Path(rec.path).name for rec in records)
            problems.append(f'Строка {row.line}: SRC_ID={row.src_id}, SRC_X={row.src_x:g}, RCV_ID={row.rcv_id}, RCV_X={row.rcv_x:g} — '+
                ('отсутствует соответствующий SGY/приёмник' if not matches else f'неоднозначное совпадение ({len(matches)} вариантов)'))
            continue
        rec,i=matches[0]
        if abs(row.src_z)>coord_tolerance or abs(row.rcv_z)>coord_tolerance:
            problems.append(f'Строка {row.line}: SRC_Z/RCV_Z должны быть нулевыми по принятому формату.')
            continue
        if row.t_ms is None:continue
        sample_float=(row.t_ms-rec.headers[i].delay_ms)*1000/rec.dt_us
        sample=int(round(sample_float))
        if not (0<=sample<rec.n_samples):
            problems.append(f'Строка {row.line}: время {row.t_ms:g} мс за пределами записи.')
            continue
        if abs(sample_float-sample)>0.25:  # 1/4 sample max roundoff
            problems.append(f'Строка {row.line}: время {row.t_ms:g} мс не совпадает с шагом {rec.dt_us} мкс.')
            continue
        wave=result[Path(rec.path).resolve()][row.wave]
        if i in wave:
            problems.append(f'Строка {row.line}: повторный пик для SRC_ID={row.src_id}, RCV_ID={row.rcv_id}, Wave={row.wave}.')
        else:
            wave[i]=sample
    if problems:
        raise ValueError('Импорт отменён. Найдены несоответствия:\n'+'\n'.join(problems[:20])+
                         (f'\n… и ещё {len(problems)-20}.' if len(problems)>20 else ''))
    return {p:dict(layers) for p,layers in result.items()}
