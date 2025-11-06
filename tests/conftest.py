import pytest

try:
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover
    TestClient = None  # type: ignore


@pytest.fixture(scope="session")
def test_client():
    if TestClient is None:
        pytest.skip("fastapi.testclient not available")
    from main import app

    with TestClient(app) as client:
        yield client
