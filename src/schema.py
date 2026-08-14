import pandera.pandas as pa
from pandera.typing import Series

class RawIAQSchema(pa.DataFrameModel):
    """
    Pandera Schema for validating raw incoming CSV sensor readings.
    Enforces exact empirical min (ge) and max (le) boundaries computed strictly on the Training Set (train_df).
    """
    Timestamp: Series[str] = pa.Field(nullable=False)

    # Environmental Features (Computed from train_df)
    Temperature: Series[float] = pa.Field(alias="Temperature (C)", ge=23.66, le=25.39, nullable=False)
    Humidity: Series[float] = pa.Field(alias="Humidity (%)", ge=33.85, le=52.83, nullable=False)
    Pressure: Series[float] = pa.Field(alias="Pressure (hPa)", ge=896.37, le=903.24, nullable=False)

    # Chemical & Air Quality Sensors
    Gas_Resistance: Series[float] = pa.Field(alias="Gas Resistance (Ohms)", ge=1060590.0, le=6334428.0, nullable=False)
    PM2_5: Series[float] = pa.Field(alias="PM2.5", ge=1.0, le=405.0, nullable=True)  # Raw nulls allowed before interpolation
    TVOC: Series[float] = pa.Field(alias="TVOC (ppb)", ge=0.0, le=60000.0, nullable=False)
    eCO2: Series[float] = pa.Field(alias="eCO2 (ppm)", ge=400.0, le=57330.0, nullable=False)
    VOC_Index: Series[float] = pa.Field(alias="VOC Index", ge=46.0, le=500.0, nullable=False)
    MQ135: Series[float] = pa.Field(alias="MQ135 Value", ge=141.0, le=200.0, nullable=False)
    Voltage: Series[float] = pa.Field(ge=0.45, le=0.65, nullable=False)
    PPM: Series[float] = pa.Field(ge=225.87, le=627.47, nullable=False)

    class Config:
        strict = True   # Requires exact schema columns
        coerce = True   # Auto-casts compatible numerical types


class ProcessedIAQSchema(pa.DataFrameModel):
    """
    Pandera Schema for validating preprocessed datasets ready for ML model training & evaluation.
    Enforces cleaned features + derived target classification label (IAQ_Class).
    """
    Timestamp: Series[str] = pa.Field(nullable=False)

    # Environmental Features
    Temperature: Series[float] = pa.Field(alias="Temperature (C)", ge=23.66, le=25.39, nullable=False)
    Humidity: Series[float] = pa.Field(alias="Humidity (%)", ge=33.85, le=52.83, nullable=False)
    Pressure: Series[float] = pa.Field(alias="Pressure (hPa)", ge=896.37, le=903.24, nullable=False)

    # Chemical & Air Quality Sensors
    Gas_Resistance: Series[float] = pa.Field(alias="Gas Resistance (Ohms)", ge=1060590.0, le=6334428.0, nullable=False)
    PM2_5: Series[float] = pa.Field(alias="PM2.5", ge=1.0, le=405.0, nullable=False)  # Interpolated (no nulls)
    TVOC: Series[float] = pa.Field(alias="TVOC (ppb)", ge=0.0, le=60000.0, nullable=False)
    eCO2: Series[float] = pa.Field(alias="eCO2 (ppm)", ge=400.0, le=57330.0, nullable=False)
    VOC_Index: Series[float] = pa.Field(alias="VOC Index", ge=46.0, le=500.0, nullable=False)
    MQ135: Series[float] = pa.Field(alias="MQ135 Value", ge=141.0, le=200.0, nullable=False)
    Voltage: Series[float] = pa.Field(ge=0.45, le=0.65, nullable=False)
    PPM: Series[float] = pa.Field(ge=225.87, le=627.47, nullable=False)

    # Classification Model Target Label
    IAQ_Class: Series[int] = pa.Field(isin=[0, 1, 2], nullable=False)

    class Config:
        strict = True   # Requires exact schema columns including IAQ_Class
        coerce = True   # Auto-casts compatible numerical types