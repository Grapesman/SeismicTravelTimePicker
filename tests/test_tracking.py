"""Regression tests for a continuous wavelet through competing events."""
from pathlib import Path
import numpy as np
import pytest
from seismic_picker.picking import PickerState
from seismic_picker.segy import SeismicRecord, TraceHeader
from seismic_picker.tracking import follow_phase


def synthetic_record(tmp_path, missing=None):
    dt_us=500
    samples=500
    heads=[TraceHeader(i+1,7,7,0.,i*2.,dt_us,0.,1) for i in range(12)]
    sig=np.zeros((12,samples),dtype=float)
    k=np.arange(samples)
    truth={}
    for i in range(12):
        arrival=140+i*4
        truth[i]=arrival
        # Strong distractor wavelet near the target phase, offset 7 ms.
        sig[i] += 1.5 * np.exp(-.5*((k-(arrival+14))/2.8)**2)
        # Coherent true wavelet (several positive/negative lobes).
        if missing is None or i!=missing:
            sig[i] += (np.exp(-.5*((k-arrival)/2.3)**2)
                       - .55*np.exp(-.5*((k-(arrival+5))/2.4)**2))
    rec=SeismicRecord(tmp_path/'fake.sgy',heads,sig,dt_us,'<',5)
    return rec, truth


def test_tracking_between_anchors_preserves_manual_and_is_not_exported(tmp_path):
    rec, truth=synthetic_record(tmp_path)
    p=PickerState(rec)
    p.set_pick(1,truth[1]);p.set_pick(10,truth[10])
    count=p.suggest_phase(radius_ms=8,polarity='positive')
    assert count==8
    assert all(abs(p.proposals[i]-truth[i])<=1 for i in range(2,10)),p.proposals
    assert len(p.rows())==2   # proposals are NEVER exported without approval
    assert len(p.proposal_info)==8
    assert all(0<=detail.confidence<=1 for detail in p.proposal_info.values())
    assert p.accept()==8
    assert len(p.rows())==10
    assert p.picks[1]==truth[1]
    assert p.picks[10]==truth[10]
    assert not p.proposal_info


def test_failed_interval_does_not_invent_samples(tmp_path):
    rec, truth=synthetic_record(tmp_path,missing=5)
    # Completely blank gap: no extrema of required phase in the corridor.
    rec.amplitudes[5,:]=0
    out=follow_phase(rec,{1:truth[1],10:truth[10]},radius_ms=4)
    assert out.points=={}
    assert out.failed_intervals==((1,10),)


def test_tracker_validates_inputs(tmp_path):
    rec,truth=synthetic_record(tmp_path)
    with pytest.raises(ValueError,match='Коридор'):
        follow_phase(rec,{0:truth[0],11:truth[11]},0.1)
    with pytest.raises(ValueError,match='полярность'):
        follow_phase(rec,{0:truth[0],11:truth[11]},8,'invalid')
    with pytest.raises(ValueError,match='Размер'):
        follow_phase(rec,{0:truth[0],11:truth[11]},8,signals=np.ones((1,2)))
    with pytest.raises(ValueError,match='Опорная'):
        follow_phase(rec,{0:truth[0],12:truth[11]},8)


def test_positive_negative_and_multiple_anchors(tmp_path):
    rec,truth=synthetic_record(tmp_path)
    # True positive event flips polarity; negative tracking still locks a
    # coherent negative event if anchors are picked on that polarity.
    inverted=SeismicRecord(rec.path, rec.headers,-rec.amplitudes.copy(),rec.dt_us,rec.endian,rec.format_code)
    out=follow_phase(inverted,{1:truth[1],6:truth[6],10:truth[10]},radius_ms=8,polarity='negative')
    assert len(out.points)==7
    assert all(abs(out.points[i].sample - truth[i])<=1 for i in out.points)
    assert not out.failed_intervals


def test_nonmonotone_receiver_coords_uses_indices(tmp_path):
    rec,truth=synthetic_record(tmp_path)
    heads=rec.headers.copy()
    heads[4]=TraceHeader(5,7,7,0.,3.,500,0.,1)
    altered=SeismicRecord(rec.path,heads,rec.amplitudes,rec.dt_us,'<',5)
    out=follow_phase(altered,{1:truth[1],10:truth[10]},8)
    assert len(out.points)==8
    assert abs(out.points[4].sample-truth[4])<=1


def test_changing_anchor_invalidates_old_proposals(tmp_path):
    rec,truth=synthetic_record(tmp_path)
    p=PickerState(rec)
    p.set_pick(1,truth[1]);p.set_pick(10,truth[10])
    assert p.suggest_phase(8)==8
    p.set_pick(1,truth[1]+1)
    assert not p.proposals
    assert not p.proposal_info
    assert not p.failed_intervals


def test_proposals_never_written_to_csv(tmp_path):
    rec,truth=synthetic_record(tmp_path)
    p=PickerState(rec)
    p.set_pick(1,truth[1]);p.set_pick(10,truth[10])
    assert p.suggest_phase(8)==8
    file=tmp_path/'export.csv'
    p.export_csv(file,all_traces=False)
    lines=file.read_text(encoding='utf-8-sig').splitlines()
    assert len(lines)==3
    p.accept()
    p.export_csv(file,all_traces=False)
    lines=file.read_text(encoding='utf-8-sig').splitlines()
    assert len(lines)==11
