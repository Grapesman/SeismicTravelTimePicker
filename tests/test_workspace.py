from pathlib import Path
import pytest
from seismic_picker.workspace import ShotSession, SessionLibrary
from seismic_picker.segy import read_segy
from seismic_picker.tracking import ProposalDetail
from test_core import sample_sgy


def test_200_files_added_without_loading(tmp_path):
    library=SessionLibrary()
    paths=[tmp_path/f'{n:03d}.sgy' for n in range(200)]
    assert len(library.add(paths)) == 200
    assert len(library.entries)==200
    assert not hasattr(library.entries[paths[0]],'amplitudes')
    assert library.add(paths[:4])==[]


def test_workspace_roundtrip_preserves_per_file_picks_proposals_and_zoom(tmp_path):
    sg1=sample_sgy(tmp_path)
    sg2=tmp_path/'other.sgy';sg2.write_bytes(sg1.read_bytes())
    lib=SessionLibrary();lib.add([sg1,sg2])
    a,b=[lib.entries[p] for p in (sg1,sg2)]
    a.picks={0:3,2:5};a.proposals={1:4}
    a.proposal_info={1: ProposalDetail(sample=4,confidence=.63,similarity=.76,deviation_ms=.3,quality='средняя')}
    a.failed_intervals=((0,3),);a.gain=2.2;a.agc=False;a.color=False
    a.selected_trace=2;a.view=((10.0,20.0),(99.0,12.0));a.n_traces=4;a.n_samples=16;a.dt_us=250;a.file_size=sg1.stat().st_size
    b.picks={3:6};b.view=((0,8),(20,0));b.gain=.8
    target=tmp_path/'session.stp-session.json'
    lib.save(target,sg2)
    loaded,current=SessionLibrary.load(target)
    assert current==sg2
    assert loaded.entries[sg1].picks == {0:3,2:5}
    assert loaded.entries[sg2].picks == {3:6}
    assert loaded.entries[sg1].proposal_info[1].confidence == pytest.approx(.63)
    assert loaded.entries[sg1].view == ((10.,20.),(99.,12.))
    assert loaded.entries[sg1].gain == pytest.approx(2.2)
    assert loaded.entries[sg2].gain == pytest.approx(.8)
    loaded.entries[sg1].validate_record(read_segy(sg1))


def test_invalid_file_modified_after_save_rejected(tmp_path):
    f=sample_sgy(tmp_path)
    s=ShotSession(f, picks={1:4}, file_size=f.stat().st_size)
    s.validate_record(read_segy(f))
    f.write_bytes(f.read_bytes()+b'foo')
    with pytest.raises(Exception):
        s.validate_record(read_segy(f))


def test_removed_sgy_detectable(tmp_path):
    lib=SessionLibrary();nonexistent=tmp_path/'ghost.sgy';lib.add([nonexistent])
    saved=tmp_path/'session.json';lib.save(saved,nonexistent)
    other,current=SessionLibrary.load(saved)
    assert current==nonexistent
    assert not other.entries[current].path.exists()
