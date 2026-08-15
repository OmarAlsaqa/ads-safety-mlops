import os
from datetime import timedelta
from pathlib import Path
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64
from feast.value_type import ValueType

# Base data source for offline features
BASE_DIR = Path(__file__).resolve().parent.parent
SOURCE_PATH = str(BASE_DIR / "data" / "processed" / "train.parquet")

source = FileSource(
    path=SOURCE_PATH,
    timestamp_field="event_timestamp",
)

# ==============================================================================
# 1. Feast Entities (Join Keys)
# ==============================================================================
ip_entity = Entity(
    name="ip",
    join_keys=["ip"],
    value_type=ValueType.INT64,
    description="Client IP Address ID"
)

app_entity = Entity(
    name="app",
    join_keys=["app"],
    value_type=ValueType.INT64,
    description="Mobile App ID"
)

device_entity = Entity(
    name="device",
    join_keys=["device"],
    value_type=ValueType.INT64,
    description="Device Model ID"
)

channel_entity = Entity(
    name="channel",
    join_keys=["channel"],
    value_type=ValueType.INT64,
    description="Marketing / Publisher Channel ID"
)

# ==============================================================================
# 2. Feast Feature Views
# ==============================================================================
ip_features_view = FeatureView(
    name="ip_features",
    entities=[ip_entity],
    ttl=timedelta(days=7),
    schema=[
        Field(name="ip_click_count", dtype=Float32),
        Field(name="ip_unique_apps", dtype=Float32),
    ],
    online=True,
    source=source,
)

app_features_view = FeatureView(
    name="app_features",
    entities=[app_entity],
    ttl=timedelta(days=7),
    schema=[
        Field(name="app_freq", dtype=Float32),
    ],
    online=True,
    source=source,
)

channel_features_view = FeatureView(
    name="channel_features",
    entities=[channel_entity],
    ttl=timedelta(days=7),
    schema=[
        Field(name="channel_freq", dtype=Float32),
    ],
    online=True,
    source=source,
)

device_features_view = FeatureView(
    name="device_features",
    entities=[device_entity],
    ttl=timedelta(days=7),
    schema=[
        Field(name="device_freq", dtype=Float32),
    ],
    online=True,
    source=source,
)
