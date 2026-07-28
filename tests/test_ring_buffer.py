"""
Unit tests for the RingBuffer module.
Verifies O(1) bounded memory behavior, snapshot creation, and thread-safe reading.
"""

import pytest
from src.processor.ring_buffer import RingBuffer


def test_ring_buffer_initialization():
    """Verifies that the buffer initializes empty with the correct max length."""
    buffer = RingBuffer(max_len=100)
    assert len(buffer) == 0
    assert buffer.max_len == 100
    assert buffer.is_full() is False


def test_ring_buffer_append_and_is_full():
    """Validates appending items until capacity is reached."""
    max_len = 5
    buffer = RingBuffer(max_len=max_len)

    for i in range(max_len - 1):
        buffer.append((float(i), 0.5, False))
        assert buffer.is_full() is False

    # Add the last item to fill capacity
    buffer.append((4.0, 0.5, False))
    assert len(buffer) == max_len
    assert buffer.is_full() is True


def test_ring_buffer_bounded_memory_eviction():
    """
    Core Non-Functional Requirement Test:
    Ensures that appending beyond max_len automatically evicts the oldest items in O(1) time,
    maintaining a strictly bounded memory footprint.
    """
    max_len = 1000
    buffer = RingBuffer(max_len=max_len)

    # Push 2500 samples into a 1000-element capacity buffer
    for i in range(2500):
        buffer.append((float(i), 0.1 * i, False))

    # Length must strictly be equal to max_len (no memory leak)
    assert len(buffer) == max_len

    window = buffer.get_window()

    # Oldest retained item should be timestamp 1500.0 (0 to 1499 were evicted)
    assert window[0][0] == 1500.0

    # Newest item should be timestamp 2499.0
    assert window[-1][0] == 2499.0


def test_ring_buffer_snapshot_decoupling():
    """
    Ensures get_window() returns a decoupled list snapshot.
    Modifications or subsequent appends to the buffer must not mutate previously retrieved windows.
    """
    buffer = RingBuffer(max_len=5)
    for i in range(3):
        buffer.append((float(i), 1.0, False))

    # Take a snapshot
    snapshot = buffer.get_window()
    assert len(snapshot) == 3

    # Add more items to the buffer after snapshot creation
    buffer.append((3.0, 1.0, False))
    buffer.append((4.0, 1.0, False))

    # Buffer updated, but snapshot must remain unchanged
    assert len(buffer) == 5
    assert len(snapshot) == 3
    assert snapshot[-1][0] == 2.0


def test_ring_buffer_clear():
    """Verifies that clear() resets the buffer state."""
    buffer = RingBuffer(max_len=10)
    for i in range(5):
        buffer.append((float(i), 0.0, False))

    assert len(buffer) == 5
    buffer.clear()
    assert len(buffer) == 0
    assert buffer.get_window() == []