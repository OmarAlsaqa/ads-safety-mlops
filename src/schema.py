import pandera.pandas as pa
from pandera.typing import Series


class RawAdClickSchema(pa.DataFrameModel):
    """
    Pandera Schema for validating raw incoming ad click streams.
    Enforces data types and ranges based on standard ad fraud detection schemas.
    """
    ip: Series[int] = pa.Field(ge=0, nullable=False)
    app: Series[int] = pa.Field(ge=0, nullable=False)
    device: Series[int] = pa.Field(ge=0, nullable=False)
    os: Series[int] = pa.Field(ge=0, nullable=False)
    channel: Series[int] = pa.Field(ge=0, nullable=False)
    click_time: Series[str] = pa.Field(nullable=False)
    is_attributed: Series[int] = pa.Field(isin=[0, 1], nullable=False)

    class Config:
        strict = False  # Allows additional metadata columns (e.g. attributed_time)
        coerce = True   # Auto-casts numerical types if compatible


class ProcessedAdClickSchema(pa.DataFrameModel):
    """
    Pandera Schema for validating feature-engineered and graph-ready ad click records.
    Enforces engineered behavioral frequencies and temporal features before graph tensor construction.
    """
    ip: Series[int] = pa.Field(ge=0, nullable=False)
    app: Series[int] = pa.Field(ge=0, nullable=False)
    device: Series[int] = pa.Field(ge=0, nullable=False)
    os: Series[int] = pa.Field(ge=0, nullable=False)
    channel: Series[int] = pa.Field(ge=0, nullable=False)
    hour: Series[int] = pa.Field(ge=0, le=23, nullable=False)
    day: Series[int] = pa.Field(ge=1, le=31, nullable=False)

    # Behavioral Entity Frequencies & Aggregations
    ip_click_count: Series[float] = pa.Field(ge=1.0, nullable=False)
    ip_unique_apps: Series[float] = pa.Field(ge=1.0, nullable=False)
    app_freq: Series[float] = pa.Field(ge=1.0, nullable=False)
    channel_freq: Series[float] = pa.Field(ge=1.0, nullable=False)
    device_freq: Series[float] = pa.Field(ge=1.0, nullable=False)

    # Advanced Temporal Deltas & Sequences (Kaggle Insights)
    next_click_delta: Series[float] = pa.Field(ge=0.0, nullable=False)
    prev_click_delta: Series[float] = pa.Field(ge=0.0, nullable=False)
    ip_device_os_cumcount: Series[float] = pa.Field(ge=0.0, nullable=False)
    ip_app_cumcount: Series[float] = pa.Field(ge=0.0, nullable=False)

    # High-Order Cross Interactions & Diversity
    ip_hh_app_count: Series[float] = pa.Field(ge=1.0, nullable=False)
    ip_hh_device_count: Series[float] = pa.Field(ge=1.0, nullable=False)
    app_channel_count: Series[float] = pa.Field(ge=1.0, nullable=False)
    ip_unique_channels: Series[float] = pa.Field(ge=1.0, nullable=False)

    # Binary Classification Target (0: Non-attributed / spam, 1: Legitimate converted install)
    is_attributed: Series[int] = pa.Field(isin=[0, 1], nullable=False)

    class Config:
        strict = False
        coerce = True