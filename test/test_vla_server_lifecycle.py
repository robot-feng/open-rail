import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from server.core.vla_server import VLAServer


def _server(model):
    server = VLAServer.__new__(VLAServer)
    server.model = model
    server._model_reset_lock = threading.Lock()
    server._client_reset_generations = {}
    server.receive_thread = threading.Thread()
    return server


def test_reset_waits_for_inference_and_runs_on_the_inference_worker():
    entered = threading.Event()
    release = threading.Event()
    second_started = threading.Event()
    calls = []
    reset_threads = []

    class Model:
        def reset(self):
            calls.append("reset")
            reset_threads.append(threading.get_ident())

        def infer(self, sequence):
            if sequence == [0]:
                calls.append("infer-start")
                entered.set()
                assert release.wait(2)
                calls.append("infer-end")
            else:
                calls.append("next-infer")
            return sequence

    server = _server(Model())

    def next_infer():
        second_started.set()
        return server._infer_model([1], b"robot", {"reset_generation": 1})

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(server._infer_model, [0], b"robot", {"reset_generation": 0})
        try:
            assert entered.wait(2)
            second = pool.submit(next_infer)
            assert second_started.wait(2)
            assert calls == ["reset", "infer-start"]
        finally:
            release.set()
        assert first.result(timeout=2) == [0]
        assert second.result(timeout=2) == [1]

    assert calls == ["reset", "infer-start", "infer-end", "reset", "next-infer"]
    assert all(ident != threading.get_ident() for ident in reset_threads)
    with pytest.raises(ValueError, match="Stale reset_generation"):
        server._infer_model([0], b"robot", {"reset_generation": 0})


def test_stateless_models_keep_concurrent_inference():
    barrier = threading.Barrier(2, timeout=2)

    class Model:
        def infer(self, sequence):
            barrier.wait()
            return sequence

    server = _server(Model())
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [
            pool.submit(server._infer_model, [value], b"robot", {"reset_generation": 0})
            for value in range(2)
        ]
        assert [result.result(timeout=3) for result in results] == [[0], [1]]


def test_close_drains_inference_before_closing_model_and_transport():
    entered = threading.Event()
    release = threading.Event()
    events = []

    class Model:
        def infer(self, sequence):
            entered.set()
            assert release.wait(2)
            events.append("inferred")

        def close(self):
            events.append("model-closed")

    server = _server(Model())
    server.thread_lock = threading.Lock()
    server.running = True
    server.executor = ThreadPoolExecutor(max_workers=1)
    server.image_decode_executor = ThreadPoolExecutor(max_workers=1)
    server.zmq_server = SimpleNamespace(close=lambda: events.append("transport-closed"))
    future = server.executor.submit(server._infer_model, [], b"robot", {})
    closer = threading.Thread(target=server.close)
    try:
        assert entered.wait(2)
        closer.start()
    finally:
        release.set()
    closer.join(timeout=3)
    assert not closer.is_alive()
    future.result(timeout=1)
    assert events == ["inferred", "model-closed", "transport-closed"]
    with pytest.raises(RuntimeError, match="shutdown"):
        server.image_decode_executor.submit(lambda: None)


@pytest.mark.parametrize("generation", [-1, True, False, 0.5, "1", None])
def test_invalid_reset_metadata_never_reaches_the_model(generation):
    calls = []
    model = SimpleNamespace(reset=lambda: calls.append("reset"), infer=lambda data: calls.append(data))
    server = _server(model)
    with pytest.raises(ValueError, match="Invalid reset_generation"):
        server._infer_model([], b"robot", {"reset_generation": generation})
    assert calls == []


@pytest.mark.parametrize("metadata", [None, {}, {"request_id": "legacy-client"}])
def test_requests_without_generation_keep_the_legacy_contract(metadata):
    calls = []
    server = _server(SimpleNamespace(
        reset=lambda: calls.append("reset"),
        infer=lambda data: calls.append(data) or "ok",
    ))
    assert server._infer_model([1], b"robot", metadata) == "ok"
    assert calls == [[1]]


def test_failed_reset_is_retried_before_any_inference():
    calls = []

    class Model:
        def reset(self):
            calls.append("reset")
            if len(calls) == 1:
                raise ConnectionError("policy unavailable")

        def infer(self, sequence):
            calls.append("infer")
            return sequence

    server = _server(Model())
    with pytest.raises(ConnectionError, match="policy unavailable"):
        server._infer_model([1], b"robot", {"reset_generation": 0})
    assert server._infer_model([2], b"robot", {"reset_generation": 0}) == [2]
    assert calls == ["reset", "reset", "infer"]


def test_parallel_requests_reset_once_for_each_generation():
    calls = []
    active = 0

    class Model:
        def reset(self):
            assert active == 0
            calls.append("reset")

        def infer(self, sequence):
            nonlocal active
            active += 1
            assert active == 1
            calls.append(sequence)
            active -= 1
            return sequence

    server = _server(Model())
    with ThreadPoolExecutor(max_workers=8) as pool:
        for generation in range(10):
            futures = [
                pool.submit(server._infer_model, [value], b"robot", {"reset_generation": generation})
                for value in range(20)
            ]
            assert [future.result(timeout=3) for future in futures] == [[i] for i in range(20)]
    assert calls.count("reset") == 10
    assert len(calls) == 210


def test_failed_inference_does_not_repeat_a_successful_reset():
    resets = []

    def infer(sequence):
        if not sequence:
            raise ValueError("invalid observation")
        return sequence

    server = _server(SimpleNamespace(reset=lambda: resets.append(True), infer=infer))
    with pytest.raises(ValueError, match="invalid observation"):
        server._infer_model([], b"robot", {"reset_generation": 0})
    assert server._infer_model([1], b"robot", {"reset_generation": 0}) == [1]
    assert resets == [True]


def test_model_close_failure_still_closes_the_transport():
    closed = []

    def fail_close():
        raise ConnectionError("close failed")

    server = _server(SimpleNamespace(close=fail_close))
    server.thread_lock = threading.Lock()
    server.executor = ThreadPoolExecutor(max_workers=1)
    server.image_decode_executor = ThreadPoolExecutor(max_workers=1)
    server.zmq_server = SimpleNamespace(close=lambda: closed.append(True))
    with pytest.raises(ConnectionError, match="close failed"):
        server.close()
    assert closed == [True]
