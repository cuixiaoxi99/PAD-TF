from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "paper.yaml"


class ExperimentConfig:
    def __init__(self, path: str | Path = DEFAULT_CONFIG):
        self.path = Path(path).resolve()
        with self.path.open("r", encoding="utf-8") as stream:
            self.raw: dict[str, Any] = yaml.safe_load(stream)
        self.root = self.path.parents[1]

    def section(self, name: str) -> dict[str, Any]:
        return self.raw[name]

    def path_for(self, name: str) -> Path:
        return (self.root / self.raw["paths"][name]).resolve()

    def dataset(self, name: str) -> dict[str, Any]:
        item = dict(self.raw["datasets"][name])
        item["name"] = name
        item["values_path"] = self.path_for("data") / item["values"]
        item["graph_path"] = self.path_for("data") / item["graph"]
        if "node_ids" in item:
            item["node_ids_path"] = self.path_for("data") / item["node_ids"]
        return item

    def pair(self, pair_id: str) -> dict[str, Any]:
        item = dict(self.raw["pairs"][pair_id])
        item["id"] = pair_id
        return item

    @property
    def hash(self) -> str:
        payload = json.dumps(self.raw, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
