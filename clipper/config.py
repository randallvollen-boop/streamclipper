"""The runtime config (`config/clipper.yaml`)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"
CONFIG_PATH = CONFIG_DIR / "clipper.yaml"
DATA_DIR = REPO_ROOT / "data"


class ConfigError(Exception):
    """Raised when `config/clipper.yaml` is malformed."""


@dataclass(frozen=True)
class Config:
    """The parsed config, with dotted access to the raw mapping."""

    raw: dict[str, Any]
    path: Path

    def section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name)
        if not isinstance(value, dict):
            return {}
        return value

    def get(self, dotted: str, default: Any = None) -> Any:
        """`cfg.get("listen.hop_seconds", 0.5)`."""
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    # -- convenience accessors used across the slices -----------------------

    @property
    def recordings_max_gb(self) -> float:
        return float(
            self.get("recordings.recordings_max_gb", 200) or 200
        )

    @property
    def segment_seconds(self) -> int:
        return int(self.get("recordings.segment_seconds", 600) or 600)

    @property
    def recordings_root(self) -> Path:
        raw = str(self.get("recordings.root", "data/recordings"))
        path = Path(raw).expanduser()
        return path if path.is_absolute() else REPO_ROOT / path

    @property
    def hop_seconds(self) -> float:
        return float(self.get("listen.hop_seconds", 0.5) or 0.5)

    @property
    def window_seconds(self) -> float:
        return float(self.get("listen.window_seconds", 2.0) or 2.0)

    @property
    def threshold(self) -> float:
        return float(self.get("listen.threshold", 1.5) or 1.5)

    @property
    def merge_seconds(self) -> float:
        return float(self.get("listen.merge_seconds", 20.0) or 20.0)

    @property
    def top_n(self) -> int:
        return int(self.get("listen.top_n", 5) or 5)

    @property
    def whisper_model(self) -> str:
        return str(self.get("listen.whisper_model", "small.en") or "small.en")

    @property
    def whisper_compute(self) -> str:
        return str(self.get("listen.whisper_compute", "int8") or "int8")

    @property
    def panns_weights(self) -> Path:
        return Path(
            str(self.get("listen.panns_weights", "~/panns_data/Cnn14_DecisionLevelMax.pth"))
        ).expanduser()

    @property
    def panns_labels(self) -> Path:
        return Path(
            str(self.get("listen.panns_labels", "~/panns_data/class_labels_indices.csv"))
        ).expanduser()

    @property
    def min_class_score(self) -> float:
        return float(self.get("listen.min_class_score", 0.25) or 0.25)

    @property
    def sound_weights(self) -> dict[str, float]:
        weights = self.section("signals").get("sound") or {}
        return {str(k): float(v) for k, v in weights.items()}

    @property
    def loudness_settings(self) -> dict[str, Any]:
        return self.section("signals").get("loudness") or {}

    @property
    def transcript_settings(self) -> dict[str, Any]:
        return self.section("signals").get("transcript") or {}

    @property
    def chat_settings(self) -> dict[str, Any]:
        return self.section("signals").get("chat") or {}

    @property
    def kit_settings(self) -> dict[str, Any]:
        return self.section("kit") or {}

    @property
    def data_dir(self) -> Path:
        """Where the runs, the decisions and the hand-off folder live."""
        raw = str(self.get("data_dir", "data") or "data")
        path = Path(raw).expanduser()
        return path if path.is_absolute() else REPO_ROOT / path

    @property
    def review_port(self) -> int:
        """The port of the review page. It is always bound to 127.0.0.1."""
        return int(self.get("review_port", 8297) or 8297)

    @property
    def handoff_dir(self) -> Path:
        """Where the approved clips are put for the person who uploads them."""
        raw = str(self.get("handoff_dir", "data/handoff") or "data/handoff")
        path = Path(raw).expanduser()
        return path if path.is_absolute() else REPO_ROOT / path

    @property
    def account(self) -> str:
        """Which clipping account the hand-off folder belongs to."""
        return str(self.get("account", "default") or "default")


def load(path: str | Path | None = None) -> Config:
    """Load `config/clipper.yaml`. Raises `ConfigError` when it is broken."""
    config_path = Path(path) if path is not None else CONFIG_PATH
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read {config_path}: {exc}") from exc
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{config_path} is not valid YAML: {exc}") from exc
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path}: expected a mapping at the top level")
    return Config(raw=raw, path=config_path)
