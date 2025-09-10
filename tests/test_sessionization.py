import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from src.processing.trip_aggregator import _split_trips_driver

def _mk_df(ts_list, speed=5.0):
    return pd.DataFrame({
        "driver_id": ["D_001"] * len(ts_list),
        "ts": pd.to_datetime(ts_list, utc=True),
        "lat": np.linspace(37.0, 37.01, len(ts_list)),
        "lon": np.linspace(-122.0, -121.99, len(ts_list)),
        "speed_mps": [speed] * len(ts_list),
        "accel_mps2": [0.0] * len(ts_list),
        "gyro_z": [0.0] * len(ts_list),
    })

def test_split_idle_gap_and_low_speed():
    t0 = datetime(2025, 9, 9, 10, 0, 0, tzinfo=timezone.utc)
    # 6 pings 10s apart -> small trip
    ts1 = [t0 + timedelta(seconds=i*10) for i in range(6)]
    # 12-minute gap -> new trip
    ts2 = [t0 + timedelta(minutes=12) + timedelta(seconds=i*10) for i in range(6)]
    df = _mk_df(ts1 + ts2)
    ranges = _split_trips_driver(df.sort_values("ts"))
    assert len(ranges) == 2, f"expected 2 trips got {ranges}"

    # Now a long low-speed stop inside one segment causes split
    ts3 = [t0 + timedelta(minutes=30) + timedelta(seconds=i*10) for i in range(6)]
    # insert a fake long stop: same timestamps spread over 4 minutes with low speed
    stop = [t0 + timedelta(minutes=25) + timedelta(seconds=i*60) for i in range(4)]
    df2 = _mk_df(ts1 + stop + ts3)
    df2.loc[len(ts1):len(ts1)+len(stop)-1, "speed_mps"] = 0.2
    ranges2 = _split_trips_driver(df2.sort_values("ts"))
    assert len(ranges2) >= 2
