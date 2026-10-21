"""Quality-control checks on actual field data (if fixture exists) and synthetic shots."""
from pathlib import Path
import numpy as np
from seismic_picker.qc import audit_picks, check_phase_continuity
from seismic_picker.segy import SeismicRecord, TraceHeader, read_segy


def _record(tmp_path):
    heads=[TraceHeader(i+1,1,1,0.0,i*2.,250,0.,1) for i in range(6)]
    ampl=np.zeros((6,1200))
    ampl[:,100:110]=1
    ampl[3,199]=1
    ampl[4,240]=-1
    ampl[5,251]=1
    return SeismicRecord(tmp_path/'fake.sgy',heads,ampl,250,'<',5)


def test_no_warning_for_gradual_progression(tmp_path):
    record=_record(tmp_path)
    picks={0:100,1:110,2:120}
    assert check_phase_continuity(record,picks,3,130,8.0)==[]
    assert audit_picks(record,{0:100,1:110,2:120,3:130}) == {}


def test_large_phase_jump_warns_and_does_not_change_points(tmp_path):
    rec=_record(tmp_path)
    existing={0:100,1:110,2:120}
    snapshot=existing.copy()
    warnings=check_phase_continuity(rec,existing,3,199,8.0)
    assert warnings
    assert warnings[0].deviation_ms > 8
    assert existing==snapshot
    assert 3 in audit_picks(rec,{**existing,3:199})


def test_polarity_switch_warning(tmp_path):
    rec=_record(tmp_path)
    picks={3:199}
    found=check_phase_continuity(rec,picks,4,240,8.,check_polarity=True)
    assert any('Полярность' in x.message for x in found)


def test_normal_large_gap_does_not_trigger_single_anchor_warning(tmp_path):
    rec=_record(tmp_path)
    assert not check_phase_continuity(rec,{0:100},5,251)


def test_real_export_has_expected_four_times():
    root=Path(__file__).resolve().parents[2]
    sgy=root/'zoo_short_2.2_X001_0001.sgy'
    if not sgy.exists():
        return
    rec=read_segy(sgy)
    from seismic_picker.picking import PickerState
    p=PickerState(rec)
    for trace,tt in [(16,259.50),(17,262.25),(18,268.25),(19,286.75)]:
        p.set_pick(trace,round(tt*1000/rec.dt_us))
    warnings=audit_picks(rec,p.picks)
    assert 19 in warnings
    assert len(p.rows())==4
    assert p.rows()[-1][6]==286.75
