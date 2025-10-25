from pathlib import Path

import pytest

from src.utils import clone_config, get_by_dotted_path, load_config, set_by_dotted_path


def test_load_config_reads_yaml(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("foo: bar\nnested:\n  value: 1\n", encoding="utf-8")

    config = load_config(config_file)

    assert config["foo"] == "bar"
    assert config["nested"]["value"] == 1


def test_load_config_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_config(Path("does_not_exist.yaml"))


def test_clone_and_set_by_dotted_path() -> None:
    original = {"training": {"learning_rate": 0.001}}
    cloned = clone_config(original)

    set_by_dotted_path(cloned, "training.learning_rate", 0.01)
    set_by_dotted_path(cloned, "training.optimizer", "adamw")

    assert original["training"]["learning_rate"] == 0.001  # original untouched
    assert cloned["training"]["learning_rate"] == 0.01
    assert get_by_dotted_path(cloned, "training.optimizer") == "adamw"
