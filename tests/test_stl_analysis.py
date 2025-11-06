import io

from fastapi.testclient import TestClient


def _get_app_module():
    # Import lazily so env vars can affect behavior per test
    import main

    return main


def _client():
    main = _get_app_module()
    return TestClient(main.app)


def _make_box_stl_bytes():
    import trimesh

    mesh = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
    buf = io.BytesIO()
    mesh.export(file_obj=buf, file_type="stl")
    return buf.getvalue(), len(mesh.faces)


def test_materials_endpoint_smoke():
    # Avoid reliance on filesystem by injecting a minimal materials DB
    import main  # noqa: E402

    main.materials_database = {"PLA": {"common": {}, "fea": {}}}
    client = TestClient(main.app)
    resp = client.get("/materials")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)
    assert "PLA" in data


def test_analyze_stl_basic_box():
    client = _client()
    stl_bytes, faces = _make_box_stl_bytes()
    files = {
        "file": ("box.stl", io.BytesIO(stl_bytes), "application/sla"),
    }
    resp = client.post("/analyze_stl", files=files)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["filename"] == "box.stl"
    assert "mesh_quality" in data
    mq = data["mesh_quality"]
    assert mq["face_count"] == faces
    assert mq["vertex_count"] > 0
    assert "mesh_quality_score" in mq


def test_analyze_stl_respects_size_limit(monkeypatch):
    # Set a very small size limit (1 KB) and ensure request is rejected
    monkeypatch.setenv("MAX_STL_MB", "0")
    client = _client()
    # Use any non-empty bytes; content is validated only for size here
    too_big = b"x" * 2048
    files = {
        "file": ("huge.stl", io.BytesIO(too_big), "application/sla"),
    }
    resp = client.post("/analyze_stl", files=files)
    assert resp.status_code == 413
    detail = resp.json().get("detail")
    # detail may be dict or string; handle dict path
    if isinstance(detail, dict):
        assert detail.get("code") == "FILE_TOO_LARGE"


def test_analyze_stl_respects_triangle_limit(monkeypatch):
    # Set triangle limit below the box triangle count to trigger 413
    monkeypatch.setenv("MAX_STL_TRIANGLES", "5")
    client = _client()
    stl_bytes, faces = _make_box_stl_bytes()
    assert faces > 5
    files = {
        "file": ("complex.stl", io.BytesIO(stl_bytes), "application/sla"),
    }
    resp = client.post("/analyze_stl", files=files)
    assert resp.status_code == 413
    detail = resp.json().get("detail")
    if isinstance(detail, dict):
        assert detail.get("code") == "MESH_TOO_COMPLEX"
