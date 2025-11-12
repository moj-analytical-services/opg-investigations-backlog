# tests/test_schema.py
import pandera as pa
import pandera.typing as pat
import pandas as pd

class IngestSchema(pa.SchemaModel):
    case_no: pat.Series[int]
    team: pat.Series[str]
    weighting: pat.Series[float] = pa.Field(ge=0, le=5)
    date_received_opg: pat.Series[pd.Timestamp]

def test_schema(raw_df: pd.DataFrame):
    IngestSchema.validate(raw_df)
