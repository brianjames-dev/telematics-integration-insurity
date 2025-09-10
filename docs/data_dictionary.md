# Telematics Data Dictionary (v1)

## 1. Raw Ping (short-term retention)

**Partition**: driver_id, date  
**Retention**: 30–60 days

| Field          | Type  | Description                          |
| -------------- | ----- | ------------------------------------ |
| driver_id      | str   | Pseudonymous ID                      |
| trip_hint_id   | str?  | Optional rolling trip ID from device |
| ts             | ts    | Timestamp (UTC)                      |
| lat, lon       | float | GPS coords (coarse grid later)       |
| gps_acc_m      | float | GPS accuracy (m)                     |
| speed_mps      | float | Instantaneous speed (m/s)            |
| accel_mps2     | float | Longitudinal acceleration (m/s²)     |
| gyro_z         | float | Yaw rate (rad/s)                     |
| source         | enum  | `phone` / `obd`                      |
| ingest_id      | uuid  | Unique ID for deduplication          |
| schema_version | int   | Schema versioning                    |

---

## 2. Trip Metadata

**Partition**: driver_id  
**Schema**:

| Field          | Type  | Description               |
| -------------- | ----- | ------------------------- |
| driver_id      | str   | Pseudonymous ID           |
| trip_id        | str   | Unique trip identifier    |
| start_ts       | ts    | Trip start time           |
| end_ts         | ts    | Trip end time             |
| duration_min   | float | Trip duration in minutes  |
| distance_km    | float | Trip distance (km)        |
| start_grid     | str   | Grid cell start           |
| end_grid       | str   | Grid cell end             |
| ping_count     | int   | # of raw pings in trip    |
| session_method | str   | Rule used to segment trip |

---

## 3. Trip Features (canonical v1)

**Partition**: driver_id, date  
**Schema**:

| Field                  | Type  | Description                               |
| ---------------------- | ----- | ----------------------------------------- |
| driver_id              | str   | Pseudonymous ID                           |
| trip_id                | str   | Unique trip ID                            |
| exposure_km            | float | Distance driven                           |
| duration_min           | float | Duration                                  |
| night_ratio            | float | Share of time 22:00–05:00                 |
| urban_km               | float | Km in urban roads                         |
| highway_km             | float | Km in highway roads                       |
| mean_speed_kph         | float | Avg trip speed                            |
| p95_over_limit_kph     | float | 95th percentile overspeed above limit     |
| pct_time_over_limit_10 | float | % of time >10kph above limit              |
| harsh_brakes_per100km  | float | Harsh braking events normalized           |
| harsh_accels_per100km  | float | Harsh acceleration events normalized      |
| cornering_per100km     | float | High yaw rate events normalized           |
| phone_motion_ratio     | float | Share of trip with phone movement (proxy) |
| rain_flag              | int   | 1 if rain during trip                     |
| crash_grid_index       | float | Local crash risk index                    |
| theft_grid_index       | float | Local theft risk index                    |
| feature_version        | int   | Feature schema version                    |

---

## 4. Driver Daily Aggregates

**Partition**: driver_id, date

| Field           | Type  | Description                      |
| --------------- | ----- | -------------------------------- |
| driver_id       | str   | Pseudonymous ID                  |
| date            | date  | Calendar date                    |
| exposure_km     | float | Daily km driven                  |
| trips           | int   | # trips that day                 |
| avg_night_ratio | float | Avg night ratio across trips     |
| events_per100km | float | Normalized event count per 100km |
| ewma_input      | float | EWMA risk input (for smoothing)  |
