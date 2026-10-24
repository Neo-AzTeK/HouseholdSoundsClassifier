"""Test suite verifying CLI runners, argument parsing, and application lifecycles."""

import os
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from household_sounds.dataset import main as dataset_main
from household_sounds.predict import main as predict_main
from household_sounds.thingy_bridge import ThingyBleManager, main as bridge_main
from household_sounds.train import main as train_main
from household_sounds.web.app import app, ble_manager, main as web_main


def test_sound_web_help():
    """Verify sound-web --help outputs help and exits with code 0."""
    with pytest.raises(SystemExit) as exc:
        web_main(["--help"])
    assert exc.value.code == 0


def test_sound_web_runner_arguments():
    """Verify sound-web configures uvicorn.run arguments and environment flags correctly."""
    with patch("uvicorn.run") as mock_uvicorn:
        # Standard launch without BLE
        web_main(["--port", "8085", "--host", "0.0.0.0"])
        assert mock_uvicorn.called
        call_kwargs = mock_uvicorn.call_args.kwargs
        assert call_kwargs["port"] == 8085
        assert call_kwargs["host"] == "0.0.0.0"
        assert "SOUND_WEB_AUTO_BLE" not in os.environ

        # Launch with --ble flag
        mock_uvicorn.reset_mock()
        web_main(["--ble", "--port", "9000"])
        assert mock_uvicorn.called
        assert os.environ.get("SOUND_WEB_AUTO_BLE") == "1"
        assert mock_uvicorn.call_args.kwargs["port"] == 9000

        # Clean up
        os.environ.pop("SOUND_WEB_AUTO_BLE", None)


def test_sound_web_lifespan_ble_startup_and_shutdown():
    """Verify that FastAPI lifespan automatically starts and stops BLE manager without error."""
    os.environ["SOUND_WEB_AUTO_BLE"] = "1"
    try:
        # TestClient enters lifespan startup on enter, and lifespan shutdown on exit
        with TestClient(app) as client:
            resp = client.get("/health")
            assert resp.status_code == 200
            # BLE manager should have been started in the running loop
            st = ble_manager.get_status()
            assert st["state"] in ("scanning", "connecting", "disconnected", "connected")

        # After client context manager exits, lifespan shutdown ran
        final_st = ble_manager.get_status()
        assert final_st["state"] == "disconnected"
    finally:
        os.environ.pop("SOUND_WEB_AUTO_BLE", None)


def test_ble_manager_missing_loop_guard():
    """Verify that calling ble_manager.start() without a running event loop does not crash."""
    mgr = ThingyBleManager()
    # Call directly outside any async loop
    mgr.start()
    assert mgr.error == "No running asyncio event loop"
    assert mgr._task is None


def test_other_project_cli_runners_help():
    """Verify all project CLI entrypoints parse --help cleanly."""
    for name, main_fn in [
        ("predict", predict_main),
        ("thingy_bridge", bridge_main),
        ("train", train_main),
        ("dataset", dataset_main),
    ]:
        with pytest.raises(SystemExit) as exc:
            main_fn(["--help"])
        assert exc.value.code == 0, f"CLI {name} --help failed with code {exc.value.code}"
