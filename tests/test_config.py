from pathlib import Path

import pytest

from src.utils import load_config


def test_load_config_reads_yaml(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("foo: bar\nnested:\n  value: 1\n", encoding="utf-8")

    config = load_config(config_file)

    assert config["foo"] == "bar"
    assert config["nested"]["value"] == 1


def test_load_config_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_config(Path("does_not_exist.yaml"))

