import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .chunker import *

def test_basic_functionality_chunk_size_1():
    """Test basic case with chunk_size=1"""
    df = pd.DataFrame({
        'timestamp': [
            datetime(2023, 1, 1, 0, 0), 
            datetime(2023, 1, 1, 0, 0),
            datetime(2023, 1, 1, 0, 1),
            datetime(2023, 1, 1, 0, 1),
            datetime(2023, 1, 1, 0, 2)
        ],
        'data': [1, 2, 3, 4, 5]
    })
    
    chunks = split_into_chunks(df, chunk_size=1)
    
    # Should create 3 chunks for 3 unique timestamps
    assert len(chunks) == 3
    
    # Verify chunk contents
    assert len(chunks[0]) == 2
    assert chunks[0].timestamp.nunique() == 1
    assert chunks[0].timestamp.iloc[0] == datetime(2023, 1, 1, 0, 0)
    
    assert len(chunks[1]) == 2
    assert chunks[1].timestamp.iloc[0] == datetime(2023, 1, 1, 0, 1)
    
    assert len(chunks[2]) == 1
    assert chunks[2].timestamp.iloc[0] == datetime(2023, 1, 1, 0, 2)

def test_chunk_size_larger_than_unique_timestamps():
    """Test when chunk_size exceeds number of unique timestamps"""
    df = generate_test_data(num_rows=100, max_duplicates=3)
    unique_ts = df.timestamp.nunique()
    
    chunks = split_into_chunks(df, chunk_size=unique_ts + 2)
    
    # Should create 1 chunk containing all data
    assert len(chunks) == 1
    pd.testing.assert_frame_equal(chunks[0], df)

def test_chunk_size_exact_divisor():
    """Test when chunk_size exactly divides unique timestamps"""
    timestamps = [datetime(2023, 1, 1, i) for i in range(6)]
    df = pd.DataFrame({
        'timestamp': np.repeat(timestamps, 3),
        'data': range(18)
    })
    
    chunks = split_into_chunks(df, chunk_size=2)
    
    # 6 unique timestamps / chunk_size 2 = 3 chunks
    assert len(chunks) == 3
    
    # Each chunk should contain 2 unique timestamps × 3 duplicates = 6 rows
    for chunk in chunks:
        assert len(chunk) == 6
        assert chunk.timestamp.nunique() == 2

def test_empty_dataframe():
    """Test handling of empty input"""
    df = pd.DataFrame(columns=['timestamp', 'data'])
    chunks = split_into_chunks(df)
    assert len(chunks) == 0

def test_order_preservation():
    """Test preservation of original timestamp order"""
    timestamps = [
        datetime(2023, 1, 1, 0, 2),
        datetime(2023, 1, 1, 0, 1),  # Earlier time appears later
        datetime(2023, 1, 1, 0, 2),
        datetime(2023, 1, 1, 0, 0)
    ]
    df = pd.DataFrame({'timestamp': timestamps})
    
    chunks = split_into_chunks(df, chunk_size=1)
    
    # Should preserve original appearance order: 0:00:02, 0:00:01, 0:00:00
    assert chunks[0].timestamp.iloc[0] == datetime(2023, 1, 1, 0, 2)
    assert chunks[1].timestamp.iloc[0] == datetime(2023, 1, 1, 0, 1)
    assert chunks[2].timestamp.iloc[0] == datetime(2023, 1, 1, 0, 0)

@pytest.mark.parametrize("chunk_size, expected_chunks, frequency", [
    (2, 5, 10),  # 4 unique timestamps / 2 = 2 chunks
    (3, 3, 10),  # 4/3 = 2 chunks (ceil)
    (5, 2, 10),  # All in one chunk
])
def test_various_chunk_sizes(chunk_size, expected_chunks, frequency):
    """Test different chunk_size configurations"""
    df = generate_test_data(num_rows=40, max_duplicates=3, num_columns=2, time_frequency=frequency)
    chunks = split_into_chunks(df, chunk_size=chunk_size)
    assert abs(len(chunks) - expected_chunks) <= 1

def test_no_timestamp_overlap():
    """Verify no timestamps appear in multiple chunks"""
    df = generate_test_data(num_rows=1000, max_duplicates=5)
    chunks = split_into_chunks(df, chunk_size=3)
    
    seen_timestamps = set()
    for chunk in chunks:
        chunk_ts = set(chunk.timestamp.unique())
        assert len(chunk_ts & seen_timestamps) == 0
        seen_timestamps.update(chunk_ts)

def test_all_rows_accounted():
    """Verify no data loss during splitting"""
    df = generate_test_data(num_rows=1000, max_duplicates=3)
    chunks = split_into_chunks(df, chunk_size=2)
    
    # Concatenate chunks and compare with original
    reconstructed = pd.concat(chunks).sort_index()
    pd.testing.assert_frame_equal(reconstructed, df)

def test_invalid_chunk_size():
    """Test error handling for invalid chunk_size"""
    df = generate_test_data(num_rows=10)
    
    with pytest.raises(ValueError):
        split_into_chunks(df, chunk_size=0)
    
    with pytest.raises(ValueError):
        split_into_chunks(df, chunk_size=-1)

def test_chunk_integrity():
    """Verify all instances of a timestamp stay in one chunk"""
    ts = datetime(2023, 1, 1, 0, 0)
    df = pd.DataFrame({
        'timestamp': [ts] * 5 + [datetime(2023, 1, 1, 0, 1)] * 3,
        'data': range(8)
    })
    
    chunks = split_into_chunks(df, chunk_size=1)
    
    # First chunk should contain all 5 instances of first timestamp
    assert len(chunks[0]) == 5
    assert chunks[0].timestamp.nunique() == 1
    assert chunks[0].timestamp.iloc[0] == ts

# Helper to run chunk inspection
def print_chunk_info(chunks):
    """Debug helper to print chunk details"""
    _ = [print(f"Chunk {i} | TS: {chunk.timestamp.unique()} | Rows: {len(chunk)}")
         for i, chunk in enumerate(chunks)]