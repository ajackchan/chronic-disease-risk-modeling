from pathlib import Path

from chronic_disease_risk.data_sources.nhanes_download import build_download_manifest, download_file, download_from_config


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


def test_download_from_config_builds_and_downloads_manifest(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "nhanes.yaml"
    config_path.write_text(
        """
base_url: https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public
cycles:
  - name: 2017-2018
    path: '2017'
    suffix: J
    files:
      - table: DEMO
""".strip(),
        encoding="utf-8",
    )

    seen: list[tuple[str, Path]] = []

    def fake_download(url: str, destination: Path) -> Path:
        seen.append((url, destination))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("ok", encoding="utf-8")
        return destination

    monkeypatch.setattr("chronic_disease_risk.data_sources.nhanes_download.download_file", fake_download)

    written = download_from_config(config_path=config_path, repo_root=tmp_path)

    assert len(written) == 1
    assert seen[0][0].endswith("/2017/DataFiles/DEMO_J.xpt")
    assert seen[0][1] == tmp_path / "data" / "raw" / "nhanes" / "2017-2018" / "DEMO_J.xpt"
