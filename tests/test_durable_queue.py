"""
Unit tests for the DurableQueue module.
Verifies file-based persistence, FIFO ordering, atomic writes,
recovery across process restarts, and corrupted file quarantine logic.
"""

import gzip
import json
import pytest
from pathlib import Path
from src.sync.durable_queue import DurableQueue


@pytest.fixture
def temp_queue_dir(tmp_path):
    """Provides a temporary directory path for queue testing."""
    return tmp_path / "test_queue"


def test_queue_push_and_length(temp_queue_dir):
    """Verifies pushing items creates compressed files and updates length correctly."""
    queue = DurableQueue(queue_dir=temp_queue_dir)
    assert len(queue) == 0

    payload_1 = {"timestamp": 100.0, "rms": 1.2}
    payload_2 = {"timestamp": 101.0, "rms": 2.5}

    batch_id_1 = queue.push(payload_1)
    batch_id_2 = queue.push(payload_2)

    assert len(queue) == 2
    assert batch_id_1.endswith(".json.gz")
    assert batch_id_2.endswith(".json.gz")
    assert (temp_queue_dir / batch_id_1).exists()
    assert (temp_queue_dir / batch_id_2).exists()


def test_queue_fifo_peek_and_pop(temp_queue_dir):
    """Validates First-In-First-Out (FIFO) ordering for peeking and popping."""
    queue = DurableQueue(queue_dir=temp_queue_dir)

    payload_first = {"seq": 1, "data": "first_batch"}
    payload_second = {"seq": 2, "data": "second_batch"}

    batch_id_1 = queue.push(payload_first)
    batch_id_2 = queue.push(payload_second)

    # Peek should return the oldest item (first_batch) without deleting it
    peek_id, peek_payload = queue.peek()
    assert peek_id == batch_id_1
    assert peek_payload["data"] == "first_batch"
    assert len(queue) == 2  # Length must remain 2 after peek

    # Pop should remove the first item from disk
    success = queue.pop(batch_id_1)
    assert success is True
    assert len(queue) == 1
    assert not (temp_queue_dir / batch_id_1).exists()

    # Next peek should return the second item
    next_id, next_payload = queue.peek()
    assert next_id == batch_id_2
    assert next_payload["data"] == "second_batch"


def test_queue_persistence_across_process_restarts(temp_queue_dir):
    """
    Simulates an application crash/restart.
    Verifies that a new queue instance pointing to the same folder 
    recovers unsent batches in chronological order.
    """
    # Phase 1: Process A writes batches to disk
    queue_instance_a = DurableQueue(queue_dir=temp_queue_dir)
    payload_a = {"timestamp": 1000.0, "event": "before_crash_1"}
    payload_b = {"timestamp": 1001.0, "event": "before_crash_2"}

    id_a = queue_instance_a.push(payload_a)
    id_b = queue_instance_a.push(payload_b)
    assert len(queue_instance_a) == 2

    # Simulate Process Crash (destroy object instance A)
    del queue_instance_a

    # Phase 2: Process B starts up (recovery phase)
    queue_instance_b = DurableQueue(queue_dir=temp_queue_dir)
    assert len(queue_instance_b) == 2

    # Verify recovering exact content
    recovered_id, recovered_payload = queue_instance_b.peek()
    assert recovered_id == id_a
    assert recovered_payload["event"] == "before_crash_1"


def test_queue_corrupted_file_quarantine(temp_queue_dir):
    """Ensures corrupted/broken files are automatically moved to quarantine to unblock queue."""
    queue = DurableQueue(queue_dir=temp_queue_dir)

    # Push a valid batch
    valid_payload = {"status": "ok"}
    valid_id = queue.push(valid_payload)

    # Create a broken/corrupted file manually in the queue directory
    corrupted_file = temp_queue_dir / "batch_0000000000000000000.json.gz"
    with open(corrupted_file, "wb") as f:
        f.write(b"NOT_A_VALID_GZIP_OR_JSON_CONTENT")

    # Queue length includes both files initially
    assert len(queue) == 2

    # Peek should encounter the corrupted file first (due to timestamps sorting)
    peek_id, peek_payload = queue.peek()

    # The queue should handle corruption gracefully: return None and quarantine the file
    assert peek_id is None
    assert peek_payload is None

    # Corrupted file should be moved to 'quarantine' directory
    quarantine_dir = temp_queue_dir / "quarantine"
    assert quarantine_dir.exists()
    assert (quarantine_dir / corrupted_file.name).exists()

    # Subsequent peek should safely return the valid file
    valid_peek_id, valid_peek_payload = queue.peek()
    assert valid_peek_id == valid_id
    assert valid_peek_payload["status"] == "ok"


def test_queue_empty_behavior(temp_queue_dir):
    """Validates behavior when operations are performed on an empty queue."""
    queue = DurableQueue(queue_dir=temp_queue_dir)

    assert len(queue) == 0
    batch_id, payload = queue.peek()
    assert batch_id is None
    assert payload is None
    assert queue.pop("non_existent_file.json.gz") is False