"""CPU-only checks for the Nsight correlation/intersection calculation."""
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("device_start,expected_overlap", [(25, 27), (70, 0)])
def test_correlated_copy_intersection(tmp_path, device_start, expected_overlap):
    database = tmp_path / "trace.sqlite"
    connection = sqlite3.connect(database)
    connection.executescript("""
        CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,text TEXT,globalTid INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(
            start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER,nameId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(
            start INTEGER,end INTEGER,bytes INTEGER,copyKind INTEGER,correlationId INTEGER);
        CREATE TABLE StringIds(id INTEGER,value TEXT);
    """)
    connection.executemany("INSERT INTO NVTX_EVENTS VALUES(?,?,?,?)", [
        (0, 10, "m12.cpu_gather.layer0", 77),
        (10, 20, "m12.h2d_pack.layer0", 77),
        (22, 52, "m12.cpu_gather.layer0", 77),
        (55, 65, "m12.h2d_pack.layer0", 77),
    ])
    connection.execute("INSERT INTO StringIds VALUES(1,'cudaMemcpyAsync_v3020')")
    connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(12,18,77,42,1)")
    connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(?,?,?,?,42)",
                       (device_start, device_start + 35, 4194304, 1))
    connection.commit()
    connection.close()
    output = tmp_path / "overlap.json"
    script = Path(__file__).resolve().parents[1] / "benchmarks/analyze_m20_overlap.py"
    subprocess.run([sys.executable, str(script), str(database), "--output", str(output)],
                   check=True, capture_output=True, text=True)
    result = json.loads(output.read_text())
    assert result["layers"] == 1
    assert result["layers_with_overlap"] == int(expected_overlap > 0)
    assert result["summed_overlap_ms"] == expected_overlap / 1e6
    assert result["layers_detail"][0]["k_h2d_bytes"] == 4194304
