"""Optional regression check against user's provided file. Never bundled with build."""
from pathlib import Path
import pytest
from seismic_picker.segy import read_segy

DATA=Path(__file__).resolve().parents[2]/'zoo_short_2.2_X001_0001.sgy'

@pytest.mark.skipif(not DATA.exists(),reason='Sample SEG-Y not checked into repository')
def test_real_trace_metadata():
    r=read_segy(DATA)
    assert r.n_traces==48
    assert r.n_samples==2048
    assert r.dt_us==250
    assert r.endian=='<'
    assert r.format_code==2
    assert [h.rcv_x for h in r.headers]==list(range(0,96,2))
    assert [h.rcv_id for h in r.headers]==list(range(1,49))
    assert all(h.src_id==1 and h.src_x==0 for h in r.headers)
    assert r.time_ms(0,400)==100.0
    assert r.time_ms(47,2047)==511.75


def test_real_phase_tracker_is_advisory_only():
    if not DATA.exists():
        pytest.skip('Нет тестового SEG-Y')
    from seismic_picker.picking import PickerState
    rec=read_segy(DATA)
    model=PickerState(rec)
    # Эти точки используются для проверки интерфейса, не как истинные вступления.
    model.set_pick(15,1000)
    model.set_pick(23,1240)
    count=model.suggest_phase(8.0,'positive')
    assert count == 7
    assert len(model.rows()) == 2
    assert model.failed_intervals == ()
    assert set(model.proposals) == set(range(16,23))
