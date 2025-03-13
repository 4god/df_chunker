import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def split_into_chunks(df: pd.DataFrame, chunk_size: int=1, timestamp_col: str='timestamp') -> list[pd.DataFrame]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if len(df) < 1:
        return df
    unique_ts, ts_indices = np.unique(df[timestamp_col].values, return_index=True)
    unique_ts = unique_ts[np.argsort(ts_indices)] # Maintain original order, to preserve time relations

    chunk_indices = np.arange(len(unique_ts)) // chunk_size
    chunk_mapper = pd.Series(chunk_indices, index=unique_ts)
    
    chunk_ids = df[timestamp_col].map(chunk_mapper).values # Map original ts to chunk ids
    
    sort_idx = np.argsort(chunk_ids) # Find sorting indexes to preserve chunk_ids's shape
    sorted_chunk_ids = chunk_ids[sort_idx] # Reorder chunk_ids according to sorting indexes
    split_points = np.where(np.diff(sorted_chunk_ids))[0] + 1 # Detect the change between chunk_ids, where new chunk starts and split there
    chunks = np.split(df.iloc[sort_idx], split_points) # Split the dataframe into chunks by sort_idx
    
    return [chunk.sort_index() for chunk in chunks] # Restore original order within chunks

def generate_test_data(
    num_rows=100,
    num_columns=3,
    start_timestamp="2025-01-01 00:00:00",
    max_duplicates=3,
    time_frequency=10
) -> pd.DataFrame:
    base_ts = [pd.Timestamp(start_timestamp) + timedelta(seconds=np.random.randint(1, time_frequency))
              for i in range(num_rows)]

    timestamps = []
    for ts in base_ts:
        timestamps.extend([ts] * np.random.randint(1, max_duplicates + 1))

    timestamps = np.random.choice(timestamps, size=num_rows)

    data = np.random.rand(num_rows, num_columns)
    columns = [f'col_{i}' for i in range(num_columns)]
    
    df = pd.DataFrame(data, columns=columns)
    df.insert(0, 'timestamp', timestamps)
    
    return df