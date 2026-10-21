from pathlib import Path
import csv
import json
import struct
import numpy as np
import pytest

from seismic_picker.segy import read_segy, SegyError, _ibm_float32
from seismic_picker.picking import PickerState, nearest_peak


def sample_sgy(tmp_path: Path, endian: str='<', scalar: int=-10) -> Path:
    b=bytearray(3600)
    struct.pack_into(endian+'H',b,3216,250)
    struct.pack_into(endian+'H',b,3220,16)
    struct.pack_into(endian+'H',b,3224,5)
    for i in range(4):
        h=bytearray(240)
        struct.pack_into(endian+'i',h,0,i+1)
        struct.pack_into(endian+'i',h,8,32)
        struct.pack_into(endian+'i',h,16,33)
        struct.pack_into(endian+'h',h,70,scalar)
        struct.pack_into(endian+'i',h,72,10000)
        struct.pack_into(endian+'i',h,80,(10000+i*20))
        struct.pack_into(endian+'h',h,108,5)
        struct.pack_into(endian+'H',h,114,16)
        struct.pack_into(endian+'H',h,116,250)
        samples=np.zeros(16,dtype=endian+'f4')
        samples[3+i]=9
        samples[6+i]=5
        b.extend(h);b.extend(samples.tobytes())
    name=tmp_path/('test_be.sgy' if endian=='>' else 'test_le.sgy')
    name.write_bytes(b)
    return name


@pytest.mark.parametrize('endian',['<','>'])
def test_load_endian_coordinates_delay(tmp_path,endian):
    record=read_segy(sample_sgy(tmp_path,endian))
    assert record.endian==endian
    assert record.n_traces==4
    assert record.n_samples==16
    assert record.dt_us==250
    assert record.headers[1].src_id==32
    assert record.headers[1].rcv_id==2
    assert record.headers[1].src_x==1000
    assert record.headers[1].rcv_x==1002
    assert record.time_ms(1,4)==6.0
    assert record.amplitudes[0,3]==9.


def test_picking_proposal_export_and_project(tmp_path):
    record=read_segy(sample_sgy(tmp_path))
    p=PickerState(record)
    p.set_pick(0,3);p.set_pick(3,6)
    assert p.suggest_between(radius_ms=.5)==2
    assert p.picks=={0:3,3:6}
    assert p.proposals=={1:4,2:5}
    assert p.rows()==[(32,1000.,0,1,1000.,0,5.75,0),(32,1000.,0,4,1006.,0,6.5,0)]
    assert p.accept()==2
    assert len(p.rows())==4
    file=tmp_path/'result.txt';p.export_txt(file)
    with file.open(encoding='utf-8-sig',newline='') as f:
        out=list(csv.reader(f,delimiter='\t'))
    assert out[0]==['SRC_ID','SRC_X','SRC_Z','RCV_ID','RCV_X','RCV_Z','Traveltime','Wave']
    assert out[1]==['32','1000','0','1','1000','0','5.75','0']
    assert out[2][-2:] == ['6','0']
    project=tmp_path/'project.json';p.save_project(project)
    restored=PickerState(record);restored.load_project(project)
    assert restored.picks==p.picks
    record.path.write_bytes(record.path.read_bytes()+b'bad')
    with pytest.raises(ValueError,match='SHA-256'):
        restored.load_project(project)


def test_empty_export_and_invalid_input(tmp_path):
    with pytest.raises(SegyError):
        read_segy(_fake_file(tmp_path))


def _fake_file(tmp_path):
    p=tmp_path/'bad.sgy';p.write_bytes(b'1234567');return p


def test_ibm_conversion():
    # IBM float 1.0: exponent 65, fraction 0x100000 => 0.0625*16 = 1
    b=bytes.fromhex('41100000')
    assert _ibm_float32(b,'>')[0] == 1.0
    assert _ibm_float32(bytes.fromhex('C1100000'),'>')[0] == -1.0
    assert _ibm_float32(b'\0\0\0\0','>')[0] == 0.0


def test_nearest_peak():
    sig=np.array([0,1,7,2,0,-1,-8,-1,0,3,6,3,0],dtype=float)
    assert nearest_peak(sig, 2.3,2,'positive')==2
    assert nearest_peak(sig, 6.3,3,'negative')==6
    assert nearest_peak(sig, 8.0,0,'positive') is None
