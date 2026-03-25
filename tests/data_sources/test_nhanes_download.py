from pathlib import Path

from chronic_disease_risk.data_sources.nhanes_download import download_file


class _Response:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        return None


def test_download_file_writes_response_content(tmp_path: Path, monkeypatch) -> None:
    destination = tmp_path / "demo.xpt"

    def fake_get(url: str, timeout: int) -> _Response:
        return _Response(b"payload")

    monkeypatch.setattr("chronic_disease_risk.data_sources.nhanes_download.requests.get", fake_get)

    result = download_file("https://example.test/demo.xpt", destination)

    assert result == destination
    assert destination.read_bytes() == b"payload"


def test_download_file_skips_existing_file(tmp_path: Path) -> None:
    existing = tmp_path / "demo.xpt"
    existing.write_bytes(b"cached")

    result = download_file("https://example.test/demo.xpt", existing)

    assert result.read_bytes() == b"cached"
