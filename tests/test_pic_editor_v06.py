"""Round-trip and validation tests for multi-wave Pick Editor."""
from pathlib import Path
from types import SimpleNamespace
import pytest

from seismic_picker.waves import WaveLayer, WaveType, valid_code, rows_for_layer
from seismic_picker.hodograph import HodoPoint, HodoSeries, geometry_relation, from_workspace
from seismic_picker.importer import read_pick_file, validate_pick_import
from seismic_picker.picking import export_table
from seismic_picker.workspace import SessionLibrary
from seismic_picker.segy import read_segy, read_segy_headers


@pytest.fixture
def sample_record(tmp_path):
    # Self-contained miniature Little-Endian SEG-Y, safe for public CI.
    import struct
    import numpy as np
    path=tmp_path/'mini.sgy'
    binary=bytearray(400)
    struct.pack_into('<H',binary,16,250)  # microsecond interval
    struct.pack_into('<H',binary,20,512)
    struct.pack_into('<H',binary,24,2)    # int32 traces
    with path.open('wb') as f:
        f.write(b' '*3200)
        f.write(binary)
        for i in range(12):
            h=bytearray(240)
            for off,value in ((0,i+1),(8,1),(16,1),(72,0),(80,i*2)):
                struct.pack_into('<i',h,off,value)
            struct.pack_into('<h',h,70,1)
            struct.pack_into('<H',h,114,512)
            struct.pack_into('<H',h,116,250)
            f.write(h)
            samples=(np.sin(np.arange(512)*.13)*10000).astype('<i4')
            f.write(samples.tobytes())
    return read_segy(path)


def test_codes_and_wave_styles():
    assert valid_code('0')=='0'
    assert valid_code('P_продольная')=='P_продольная'
    for value in ('','invalid wave','a\tb','a'*40,'a;b'):
        with pytest.raises(ValueError):valid_code(value)
    assert WaveType(code='R',name='Отраженная',line_color='#112233').code=='R'


def test_multiwave_round_trip(tmp_path,sample_record):
    library=SessionLibrary()
    p=sample_record.path
    library.add([p])
    session=library.entries[p]
    session.n_traces=sample_record.n_traces
    session.n_samples=sample_record.n_samples
    session.dt_us=sample_record.dt_us
    session.wave_layers['0']=WaveLayer(picks={5:100})
    session.wave_layers['R']=WaveLayer(picks={5:180})
    session.geometry=[{'src_id':h.src_id,'src_x':h.src_x,'rcv_id':h.rcv_id,
                       'rcv_x':h.rcv_x,'delay_ms':h.delay_ms} for h in sample_record.headers]
    library.wave_types['R']=WaveType(code='R',name='Отраженная',line_color='#345678')
    output=tmp_path/'multilayer.stp-session.json'
    library.save(output,p)
    restored,current=SessionLibrary.load(output)
    assert current==p
    assert restored.wave_types['R'].line_color=='#345678'
    assert restored.entries[p].wave_layers['0'].picks[5]==100
    assert restored.entries[p].wave_layers['R'].picks[5]==180
    assert len(from_workspace(restored))==2


def test_export_import_multiple_waves_same_receiver(tmp_path,sample_record):
    rows=(rows_for_layer(sample_record,WaveLayer(picks={7:150,9:180}),'0')+
          rows_for_layer(sample_record,WaveLayer(picks={7:225}),'P'))
    dest=tmp_path/'wave.txt'
    export_table(dest,rows)
    parsed=read_pick_file(dest)
    light=read_segy_headers(sample_record.path)
    match=validate_pick_import(parsed,[light])
    assert match[sample_record.path]['0']=={7:150,9:180}
    assert match[sample_record.path]['P']=={7:225}
    assert 'P' in dest.read_text(encoding='utf-8-sig')


def test_missing_sgy_detection(tmp_path,sample_record):
    dest=tmp_path/'waves.txt'
    export_table(dest,[(999,0,0,1,0,0,55,'0')])
    with pytest.raises(ValueError,match='отсутствует'):
        validate_pick_import(read_pick_file(dest),[read_segy_headers(sample_record.path)])


def test_duplicate_record_ambiguity(tmp_path,sample_record):
    dest=tmp_path/'waves.txt'
    export_table(dest,rows_for_layer(sample_record,WaveLayer(picks={5:120}),'0'))
    metadata=read_segy_headers(sample_record.path)
    second=SimpleNamespace(headers=metadata.headers,n_samples=metadata.n_samples,
                           dt_us=metadata.dt_us,path=tmp_path/'same_headers.sgy')
    with pytest.raises(ValueError,match='неоднозначное'):
        validate_pick_import(read_pick_file(dest),[metadata,second])


def test_duplicate_wave_receiver_is_rejected(tmp_path,sample_record):
    dest=tmp_path/'waves.txt'
    row=rows_for_layer(sample_record,WaveLayer(picks={5:120}),'R')[0]
    export_table(dest,[row,row])
    with pytest.raises(ValueError,match='повторный пик'):
        validate_pick_import(read_pick_file(dest),[read_segy_headers(sample_record.path)])


def test_reciprocal_and_overtaking():
    a=HodoSeries(Path('a.sgy'),'0',1,-10,(),(0,100))
    b=HodoSeries(Path('b.sgy'),'0',2,110,(),(0,100))
    c=HodoSeries(Path('c.sgy'),'0',3,-25,(),(0,100))
    inside=HodoSeries(Path('d.sgy'),'0',4,50,(),(0,100))
    assert geometry_relation(a,b)=='встречные'
    assert geometry_relation(a,c)=='нагоняющие'
    assert geometry_relation(a,inside)=='неоднозначная геометрия'
