from seismic_picker.picking import export_table

def test_export_combined_preserves_source_ids(tmp_path):
    destination = tmp_path / 'merged.txt'
    rows=[(1,0,0,17,32,0,259.5,0),(2,100,0,17,132,0,260.25,0)]
    export_table(destination,rows)
    lines=destination.read_text(encoding='utf-8-sig').splitlines()
    assert len(lines)==3
    assert lines[1].split('\t')==['1','0','0','17','32','0','259.5','0']
    assert lines[2].split('\t')==['2','100','0','17','132','0','260.25','0']
