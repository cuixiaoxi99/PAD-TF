from __future__ import annotations

import json
import math
import os
import platform
import socket
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Subset

from .config import ExperimentConfig
from .data import (
    DatasetBundle,
    batch_size_for,
    blocked_stratified_probe_split,
    build_dataset_bundle,
    make_loader,
)
from .decision import (
    DomainSignals,
    choose_strategy,
    combine_signals,
    predict_finetune_epochs,
    static_discrepancy,
)
from .graph import GraphBundle, build_graph_bundle, load_adjacency
from .model import (
    FlashSTTransferModel,
    load_source_into_target,
    parameter_count,
    trainable_parameter_count,
)
from .selectors import estimate_coreset_coverage, select_windows
from .trainer import (
    evaluate_model,
    fit_model,
    fit_model_milestones,
    fit_model_time_budget,
    persistence_mae,
    validation_mae,
)
from .utility import deployment_utility
from .utils import (
    atomic_write_json,
    build_logger,
    command_output,
    elapsed_timer,
    set_seed,
    sha256_file,
)


PAPER_METHODS = [
    "Direct",
    "Fixed-FT",
    "Retrain",
    "FlashST-Full",
    "Random-FT",
    "CoreSet-FT",
    "Proxy-FT",
    "ELFS-FT",
    "BADGE-FT",
    "UHerding-FT",
    "PAD-TF",
]

ANALYSIS_POLICIES = {
    "dynamic_only",
    "static_only",
    "fixed_b",
    "no_cost",
    "lambda_0.1",
    "lambda_0.2",
    "lambda_0.5",
    "lambda_1.0",
}

TIME_MATCHED_METHODS = [
    "FlashST-Full",
    "Random-FT",
    "CoreSet-FT",
    "Proxy-FT",
    "ELFS-FT",
    "BADGE-FT",
    "UHerding-FT",
]


def _method_slug(method: str) -> str:
    return method.lower().replace("-", "_")


def _probe_response_statistics(
    zero_shot_mae: float,
    history: list[dict[str, float]],
    epsilon: float,
) -> dict[str, float]:
    if len(history) < 2:
        raise ValueError("The epoch predictor requires at least two probe epochs")
    first_gain = (zero_shot_mae - float(history[0]["val_mae"])) / (zero_shot_mae + epsilon)
    second_gain = (float(history[0]["val_mae"]) - float(history[1]["val_mae"])) / (
        zero_shot_mae + epsilon
    )
    decay = second_gain / max(first_gain, epsilon)
    decay = min(max(decay, epsilon), 1.0 - epsilon)
    return {
        "first_gain": first_gain,
        "second_gain": second_gain,
        "response_decay": decay,
        "response_tau": -1.0 / math.log(decay),
    }


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig, device: str | None = None) -> None:
        self.config = config
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.training = config.section("training")
        self.data_settings = config.section("data")
        self._data_cache: dict[str, DatasetBundle] = {}
        self._graph_cache: dict[str, GraphBundle] = {}

    def data(self, name: str) -> DatasetBundle:
        if name not in self._data_cache:
            self._data_cache[name] = build_dataset_bundle(self.config, name)
        return self._data_cache[name]

    def graph(self, name: str) -> GraphBundle:
        if name not in self._graph_cache:
            self._graph_cache[name] = build_graph_bundle(self.config, name)
        return self._graph_cache[name]

    def source_checkpoint(self, dataset_name: str, seed: int) -> Path:
        return self.config.path_for("checkpoints") / "source" / dataset_name / f"seed_{seed}" / "best.pt"

    def run_checkpoint(self, pair_id: str, seed: int, method: str) -> Path:
        return self.config.path_for("checkpoints") / "runs" / pair_id / f"seed_{seed}" / f"{_method_slug(method)}.pt"

    def result_path(self, pair_id: str, seed: int, method: str) -> Path:
        return self.config.path_for("results") / "runs" / pair_id / f"seed_{seed}" / f"{_method_slug(method)}.json"

    def probe_result_path(self, pair_id: str, seed: int) -> Path:
        return self.config.path_for("results") / "calibration" / "probes" / f"{pair_id}_seed{seed}.json"

    def policy_result_path(self, pair_id: str, seed: int, policy: str) -> Path:
        return self.config.path_for("results") / "analysis" / "policies" / policy / pair_id / f"seed_{seed}.json"

    def control_result_path(self, control: str, seed: int) -> Path:
        return self.config.path_for("results") / "analysis" / "controls" / control / f"seed_{seed}.json"

    def time_matched_result_path(self, pair_id: str, seed: int, method: str) -> Path:
        return (
            self.config.path_for("results")
            / "analysis"
            / "time_matched"
            / _method_slug(method)
            / pair_id
            / f"seed_{seed}.json"
        )

    def logger(self, pair_id: str, seed: int, method: str) -> Any:
        path = self.config.path_for("logs") / pair_id / f"seed_{seed}"
        path.mkdir(parents=True, exist_ok=True)
        return build_logger(f"{pair_id}-{seed}-{method}", path / f"{_method_slug(method)}.log")

    def train_source(self, dataset_name: str, seed: int, force: bool = False) -> Path:
        checkpoint = self.source_checkpoint(dataset_name, seed)
        if checkpoint.exists() and not force:
            return checkpoint
        set_seed(seed)
        data = self.data(dataset_name)
        graph = self.graph(dataset_name)
        model = FlashSTTransferModel(self.config, dataset_name, graph, self.device, "source").to(self.device)
        batch_size = batch_size_for(self.config, dataset_name)
        train_loader = make_loader(data.train, batch_size, True, seed, int(self.data_settings["num_workers"]))
        val_loader = make_loader(data.val, batch_size, False, seed, int(self.data_settings["num_workers"]))
        logger = self.logger(f"source_{dataset_name}", seed, "source")
        logger.info("source_train dataset=%s seed=%d parameters=%d", dataset_name, seed, parameter_count(model))
        fit = fit_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            scaler=data.scaler,
            device=self.device,
            epochs=int(self.training["source_epochs"]),
            learning_rate=float(self.training["learning_rate"]),
            effective_batch_size=int(self.training["effective_batch_size"]),
            ssl_weight=float(self.training["ssl_weight"]),
            grad_clip=float(self.training["grad_clip"]),
            patience=int(self.training["source_patience"]),
            checkpoint_path=checkpoint,
            logger=logger,
        )
        payload = torch.load(checkpoint, map_location="cpu")
        payload.update(
            {
                "dataset": dataset_name,
                "seed": seed,
                "stage": "source",
                "config_hash": self.config.hash,
                "upstream_commit": self.config.section("project")["upstream_commit"],
                "fit": fit.__dict__,
            }
        )
        torch.save(payload, checkpoint)
        logger.info("source_complete checkpoint=%s sha256=%s", checkpoint, sha256_file(checkpoint))
        return checkpoint

    def _target_model(
        self,
        target: str,
        source_checkpoint: Path,
        map_pretrain_norm: bool = False,
        prompt_sign: float = 1.0,
    ) -> tuple[FlashSTTransferModel, dict[str, Any]]:
        model = FlashSTTransferModel(self.config, target, self.graph(target), self.device, "target").to(self.device)
        transfer = load_source_into_target(model, source_checkpoint, map_pretrain_norm, prompt_sign)
        return model, transfer

    def _load_result(self, pair_id: str, seed: int, method: str) -> dict[str, Any] | None:
        path = self.result_path(pair_id, seed, method)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _load_trained_model(self, model: FlashSTTransferModel, checkpoint: Path) -> None:
        payload = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(payload["state_dict"])
        model.to(self.device)

    def _base_record(
        self,
        pair_id: str,
        seed: int,
        method: str,
        source_checkpoint: Path,
        target: str,
    ) -> dict[str, Any]:
        pair = self.config.pair(pair_id)
        gpu_name = torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "cpu"
        return {
            "pair": pair_id,
            "source": pair["source"],
            "target": target,
            "method": method,
            "seed": seed,
            "backbone": self.config.section("project")["backbone"],
            "upstream_commit": self.config.section("project")["upstream_commit"],
            "config_hash": self.config.hash,
            "source_checkpoint": str(source_checkpoint),
            "source_checkpoint_sha256": sha256_file(source_checkpoint),
            "device": str(self.device),
            "gpu": gpu_name,
            "hostname": socket.gethostname(),
        }

    def run_method(self, pair_id: str, seed: int, method: str, force: bool = False) -> dict[str, Any]:
        if method not in PAPER_METHODS:
            raise ValueError(f"Unknown method: {method}")
        result_path = self.result_path(pair_id, seed, method)
        if result_path.exists() and not force:
            return json.loads(result_path.read_text(encoding="utf-8"))
        set_seed(seed)
        pair = self.config.pair(pair_id)
        source = pair["source"]
        target = pair["target"]
        source_checkpoint = self.train_source(source, seed, force=False)
        data = self.data(target)
        logger = self.logger(pair_id, seed, method)
        batch_size = batch_size_for(self.config, target)
        val_loader = make_loader(data.val, batch_size, False, seed, int(self.data_settings["num_workers"]))
        test_loader = make_loader(data.test, batch_size, False, seed, int(self.data_settings["num_workers"]))
        record = self._base_record(pair_id, seed, method, source_checkpoint, target)
        record["target_windows"] = {"train": len(data.train), "val": len(data.val), "test": len(data.test)}
        logger.info("run_start pair=%s method=%s seed=%d", pair_id, method, seed)

        if method == "Direct":
            model, transfer = self._target_model(target, source_checkpoint)
            metrics = evaluate_model(model, test_loader, data.scaler, self.device)
            record.update(
                {
                    "selected_strategy": "Direct",
                    "ratio": 0.0,
                    "epochs": 0,
                    "metrics": metrics,
                    "timing": {"selection": 0.0, "probe": 0.0, "adaptation": 0.0, "total_target": 0.0},
                    "transfer": transfer,
                }
            )
        elif method == "Retrain":
            model = FlashSTTransferModel(self.config, target, self.graph(target), self.device, "retrain").to(self.device)
            train_loader = make_loader(data.train, batch_size, True, seed, int(self.data_settings["num_workers"]))
            checkpoint = self.run_checkpoint(pair_id, seed, method)
            fit = fit_model(
                model,
                train_loader,
                val_loader,
                data.scaler,
                self.device,
                int(self.training["retrain_epochs"]),
                float(self.training["learning_rate"]),
                int(self.training["effective_batch_size"]),
                0.0,
                float(self.training["grad_clip"]),
                int(self.training["retrain_patience"]),
                checkpoint,
                logger,
            )
            metrics = evaluate_model(model, test_loader, data.scaler, self.device)
            record.update(
                {
                    "selected_strategy": "Retrain",
                    "ratio": 1.0,
                    "epochs": fit.epochs_run,
                    "best_epoch": fit.best_epoch,
                    "metrics": metrics,
                    "timing": {"selection": 0.0, "probe": 0.0, "adaptation": fit.elapsed_seconds, "total_target": fit.elapsed_seconds},
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "parameters": {"total": parameter_count(model), "trainable": trainable_parameter_count(model)},
                }
            )
        elif method == "PAD-TF":
            record.update(self._run_padtf(pair_id, seed, source_checkpoint, data, val_loader, test_loader, logger))
        else:
            model, transfer = self._target_model(target, source_checkpoint)
            selection_seconds = 0.0
            selection_metadata: dict[str, Any] = {}
            indices = None
            ratio = 1.0
            epochs = int(self.training["full_epochs"])
            if method == "Fixed-FT":
                ratio = float(self.training.get("fixed_ratio", 0.30))
                epochs = int(self.training.get("fixed_epochs", 10))
                selection = select_windows(
                    "Random-FT",
                    model,
                    data.train,
                    data.scaler,
                    ratio,
                    self.device,
                    seed,
                    self.config.section("selectors"),
                )
                indices = selection.indices
                selection_seconds = selection.elapsed_seconds
                selection_metadata = {
                    **selection.metadata,
                    "role": "fixed_budget_uniform_subset",
                }
            elif method != "FlashST-Full":
                ratio = float(self.training["selection_ratio"])
                epochs = int(self.training["selection_epochs"])
                selection = select_windows(
                    method,
                    model,
                    data.train,
                    data.scaler,
                    ratio,
                    self.device,
                    seed,
                    self.config.section("selectors"),
                )
                indices = selection.indices
                selection_seconds = selection.elapsed_seconds
                selection_metadata = selection.metadata
            train_loader = make_loader(
                data.train,
                batch_size,
                True,
                seed,
                int(self.data_settings["num_workers"]),
                indices,
            )
            checkpoint = self.run_checkpoint(pair_id, seed, method)
            fit = fit_model(
                model,
                train_loader,
                val_loader,
                data.scaler,
                self.device,
                epochs,
                float(self.training["learning_rate"]),
                int(self.training["effective_batch_size"]),
                float(self.training["ssl_weight"]),
                float(self.training["grad_clip"]),
                None,
                checkpoint,
                logger,
            )
            metrics = evaluate_model(model, test_loader, data.scaler, self.device)
            total = selection_seconds + fit.elapsed_seconds
            record.update(
                {
                    "selected_strategy": "Fine-tune",
                    "ratio": ratio,
                    "epochs": epochs,
                    "best_epoch": fit.best_epoch,
                    "metrics": metrics,
                    "timing": {"selection": selection_seconds, "probe": 0.0, "adaptation": fit.elapsed_seconds, "total_target": total},
                    "selection": selection_metadata,
                    "transfer": transfer,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "parameters": {"total": parameter_count(model), "trainable": trainable_parameter_count(model)},
                }
            )
        atomic_write_json(result_path, record)
        logger.info(
            "run_complete pair=%s method=%s rmse=%.6f total_target=%.3f result=%s",
            pair_id,
            method,
            record["metrics"]["rmse"],
            record["timing"]["total_target"],
            result_path,
        )
        return record

    def run_probe(self, pair_id: str, seed: int, force: bool = False) -> dict[str, Any]:
        set_seed(seed)
        pair = self.config.pair(pair_id)
        source = pair["source"]
        target = pair["target"]
        source_checkpoint = self.train_source(source, seed, force=False)
        source_hash = sha256_file(source_checkpoint)
        result_path = self.probe_result_path(pair_id, seed)
        if result_path.exists() and not force:
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            if existing.get("config_hash") == self.config.hash and existing.get("source_checkpoint_sha256") == source_hash:
                return existing
        data = self.data(target)
        settings = self.config.section("padtf")
        batch_size = batch_size_for(self.config, target)
        logger = self.logger(pair_id, seed, "PAD-TF-Probe")
        with elapsed_timer(self.device) as static_timer:
            static = static_discrepancy(
                self.data(source),
                data,
                self.graph(source),
                self.graph(target),
                float(settings["epsilon"]),
            )
        probe_gap = max(
            int(settings.get("probe_block_gap", 0)),
            int(self.data_settings["history"]) + int(self.data_settings["horizon"]),
        )
        probe_train, probe_val = blocked_stratified_probe_split(
            data.train,
            float(settings["probe_ratio"]),
            float(settings["probe_train_fraction"]),
            seed,
            probe_gap,
        )
        probe_train_loader = make_loader(data.train, batch_size, True, seed, indices=probe_train)
        probe_val_loader = make_loader(data.train, batch_size, False, seed, indices=probe_val)
        probe_model, transfer = self._target_model(target, source_checkpoint)
        with elapsed_timer(self.device) as zero_timer:
            zero_shot_mae = validation_mae(probe_model, probe_val_loader, data.scaler, self.device)
            naive_mae = persistence_mae(probe_val_loader, data.scaler, self.device)
        probe_checkpoint = (
            self.config.path_for("checkpoints")
            / "calibration"
            / "probes"
            / pair_id
            / f"seed_{seed}.pt"
        )
        probe_fit = fit_model(
            probe_model,
            probe_train_loader,
            probe_val_loader,
            data.scaler,
            self.device,
            int(settings["probe_epochs"]),
            float(self.training["learning_rate"]),
            int(self.training["effective_batch_size"]),
            float(self.training["ssl_weight"]),
            float(self.training["grad_clip"]),
            None,
            probe_checkpoint,
            logger,
        )
        final_probe_mae = probe_fit.best_val_mae
        probe_gain = (zero_shot_mae - final_probe_mae) / (zero_shot_mae + float(settings["epsilon"]))
        response = _probe_response_statistics(
            zero_shot_mae,
            probe_fit.history,
            float(settings["epsilon"]),
        )
        signals = combine_signals(
            static,
            zero_shot_mae,
            naive_mae,
            probe_gain,
            float(settings["alpha"]),
            float(settings["epsilon"]),
        )
        total_probe = static_timer["seconds"] + zero_timer["seconds"] + probe_fit.elapsed_seconds
        result = {
            "pair": pair_id,
            "source": source,
            "target": target,
            "seed": seed,
            "config_hash": self.config.hash,
            "source_checkpoint": str(source_checkpoint),
            "source_checkpoint_sha256": source_hash,
            "probe_checkpoint": str(probe_checkpoint),
            "probe_checkpoint_sha256": sha256_file(probe_checkpoint),
            "signals": signals.__dict__,
            "timing": {
                "static": static_timer["seconds"],
                "zero_shot": zero_timer["seconds"],
                "probe_training": probe_fit.elapsed_seconds,
                "total_probe": total_probe,
            },
            "probe": {
                "ratio": float(settings["probe_ratio"]),
                "train_windows": len(probe_train),
                "val_windows": len(probe_val),
                "epochs": int(settings["probe_epochs"]),
                "split_protocol": "temporally_blocked_stratified",
                "minimum_start_gap": probe_gap,
                "observed_start_gap": min(probe_val) - max(probe_train),
                "zero_shot_mae": zero_shot_mae,
                "final_mae": final_probe_mae,
                "naive_mae": naive_mae,
                "gain": probe_gain,
                **response,
            },
            "transfer": transfer,
        }
        atomic_write_json(result_path, result)
        return result

    def _run_padtf(
        self,
        pair_id: str,
        seed: int,
        source_checkpoint: Path,
        data: DatasetBundle,
        val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        logger: Any,
    ) -> dict[str, Any]:
        pair = self.config.pair(pair_id)
        target = pair["target"]
        settings = self.config.section("padtf")
        batch_size = batch_size_for(self.config, target)
        probe_record = self.run_probe(pair_id, seed)
        signals = DomainSignals(**probe_record["signals"])
        if "response_tau" in probe_record["probe"]:
            probe_response = {
                key: float(probe_record["probe"][key])
                for key in ("first_gain", "second_gain", "response_decay", "response_tau")
            }
        else:
            probe_checkpoint = torch.load(probe_record["probe_checkpoint"], map_location="cpu")
            probe_response = _probe_response_statistics(
                float(probe_record["probe"]["zero_shot_mae"]),
                probe_checkpoint["history"],
                float(settings["epsilon"]),
            )
        predicted_epochs, epoch_prediction = predict_finetune_epochs(
            self.config,
            pair_id,
            signals,
            probe_response["response_tau"],
        )
        retrain_result = self._load_result(pair_id, seed, "Retrain")
        retrain_seconds = None if retrain_result is None else float(retrain_result["timing"]["total_target"])
        if retrain_result is not None:
            full_epoch_seconds = retrain_seconds / max(1, int(retrain_result["epochs"]))
        else:
            probe_fraction = max(float(probe_record["probe"]["train_windows"]) / max(1, len(data.train)), 1.0e-6)
            full_epoch_seconds = (
                float(probe_record["timing"]["probe_training"])
                / max(1, int(settings["probe_epochs"]))
                / probe_fraction
            )
        coverage_model, _ = self._target_model(target, source_checkpoint)
        coverage_profile = estimate_coreset_coverage(
            coverage_model,
            data.train,
            data.scaler,
            [float(ratio) for ratio in settings["ratios"]],
            self.device,
            seed,
            self.config.section("selectors"),
        )
        del coverage_model
        selector_estimate = (
            coverage_profile.elapsed_seconds
            + 0.01 * float(retrain_seconds or full_epoch_seconds * int(self.training["retrain_epochs"]))
        )
        with elapsed_timer(self.device) as grid_timer:
            decision = choose_strategy(
                self.config,
                pair_id,
                signals,
                full_epoch_seconds,
                retrain_seconds,
                selector_estimate,
                epoch_candidates=[predicted_epochs],
                ratio_coverage=coverage_profile.values,
                signed_log_nodes=float(epoch_prediction["features"]["signed_log_nodes"]),
                flow_target=bool(epoch_prediction["features"]["flow_target"]),
            )
        logger.info(
            "padtf_decision strategy=%s ratio=%.2f epochs=%d phi=%.6f probe_gain=%.6f calibration=%s",
            decision.strategy,
            decision.ratio,
            decision.epochs,
            signals.phi,
            signals.probe_gain,
            decision.calibration.source,
        )
        selection_seconds = 0.0
        adaptation_seconds = 0.0
        selection_metadata: dict[str, Any] = {}
        epoch_metadata: dict[str, Any] = {
            "mode": "pretraining_formula",
            "predicted_epoch": predicted_epochs,
            "selected_epoch": decision.epochs,
            "probe_response": probe_response,
            **epoch_prediction,
        }
        checkpoint: Path | None = None
        set_seed(seed)
        if decision.strategy == "Direct":
            final_model, transfer = self._target_model(target, source_checkpoint)
            metrics = evaluate_model(final_model, test_loader, data.scaler, self.device)
        elif decision.strategy == "Retrain":
            retrain_result = retrain_result or self.run_method(pair_id, seed, "Retrain")
            metrics = retrain_result["metrics"]
            adaptation_seconds = float(retrain_result["timing"]["adaptation"])
            transfer = {"reused_retrain_baseline": True}
            checkpoint = Path(retrain_result["checkpoint"])
        else:
            final_model, transfer = self._target_model(target, source_checkpoint)
            selection = select_windows(
                "Budget-CoreSet-FT",
                final_model,
                data.train,
                data.scaler,
                decision.ratio,
                self.device,
                seed,
                self.config.section("selectors"),
            )
            selection_seconds = selection.elapsed_seconds
            selection_metadata = selection.metadata
            train_loader = make_loader(data.train, batch_size, True, seed, indices=selection.indices)
            checkpoint = self.run_checkpoint(pair_id, seed, "PAD-TF")
            fit = fit_model_milestones(
                final_model,
                train_loader,
                val_loader,
                data.scaler,
                self.device,
                [decision.epochs],
                float(self.training["learning_rate"]),
                int(self.training["effective_batch_size"]),
                float(self.training["ssl_weight"]),
                float(self.training["grad_clip"]),
                {decision.epochs: checkpoint},
                logger,
            )[decision.epochs]
            adaptation_seconds = fit.elapsed_seconds
            metrics = evaluate_model(final_model, test_loader, data.scaler, self.device)
            epoch_metadata["epochs_run"] = fit.epochs_run
            epoch_metadata["checkpoint_epoch"] = fit.best_epoch
        probe_seconds = float(probe_record["timing"]["total_probe"]) + grid_timer["seconds"]
        total = probe_seconds + coverage_profile.elapsed_seconds + selection_seconds + adaptation_seconds
        result: dict[str, Any] = {
            "selected_strategy": decision.strategy,
            "ratio": decision.ratio,
            "epochs": decision.epochs,
            "epoch_prediction": epoch_metadata,
            "metrics": metrics,
            "timing": {
                "static": float(probe_record["timing"]["static"]),
                "zero_shot": float(probe_record["timing"]["zero_shot"]),
                "probe_training": float(probe_record["timing"]["probe_training"]),
                "grid": grid_timer["seconds"],
                "coverage_profile": coverage_profile.elapsed_seconds,
                "selection": selection_seconds,
                "adaptation": adaptation_seconds,
                "probe": probe_seconds,
                "total_target": total,
            },
            "probe": {**probe_record["probe"], "checkpoint": probe_record["probe_checkpoint"]},
            "signals": signals.__dict__,
            "decision": decision.to_dict(),
            "ratio_coverage": {
                "values": coverage_profile.values,
                **coverage_profile.metadata,
            },
            "selection": selection_metadata,
            "transfer": transfer,
            "probe_transfer": probe_record["transfer"],
        }
        if checkpoint is not None and checkpoint.exists():
            result["checkpoint"] = str(checkpoint)
            result["checkpoint_sha256"] = sha256_file(checkpoint)
        return result

    def run_pair(
        self,
        pair_id: str,
        seed: int,
        force: bool = False,
        methods: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        records = []
        selected_methods = PAPER_METHODS if methods is None else methods
        unknown = sorted(set(selected_methods) - set(PAPER_METHODS))
        if unknown:
            raise ValueError(f"Unknown methods: {unknown}")
        for method in selected_methods:
            records.append(self.run_method(pair_id, seed, method, force=force))
        return records

    def verify_result(
        self,
        pair_id: str,
        seed: int,
        method: str,
        tolerance: float = 1.0e-4,
    ) -> dict[str, Any]:
        record = self._load_result(pair_id, seed, method)
        if record is None:
            raise FileNotFoundError(self.result_path(pair_id, seed, method))
        set_seed(seed)
        pair = self.config.pair(pair_id)
        target = pair["target"]
        source_checkpoint = self.source_checkpoint(pair["source"], seed)
        data = self.data(target)
        batch_size = batch_size_for(self.config, target)
        test_loader = make_loader(
            data.test,
            batch_size,
            False,
            seed,
            int(self.data_settings["num_workers"]),
        )
        selected_strategy = str(record["selected_strategy"])
        if selected_strategy == "Direct":
            model, _ = self._target_model(target, source_checkpoint)
        elif method == "Retrain" or selected_strategy == "Retrain":
            model = FlashSTTransferModel(self.config, target, self.graph(target), self.device, "retrain").to(self.device)
            self._load_trained_model(model, Path(record["checkpoint"]))
        else:
            model, _ = self._target_model(target, source_checkpoint)
            self._load_trained_model(model, Path(record["checkpoint"]))
        observed = evaluate_model(model, test_loader, data.scaler, self.device)
        deltas = {
            metric: abs(float(observed[metric]) - float(record["metrics"][metric]))
            for metric in ("mae", "rmse", "mape")
        }
        source_hash_matches = sha256_file(source_checkpoint) == record["source_checkpoint_sha256"]
        checkpoint_hash_matches = True
        if record.get("checkpoint") and record.get("checkpoint_sha256"):
            checkpoint_hash_matches = sha256_file(record["checkpoint"]) == record["checkpoint_sha256"]
        return {
            "pair": pair_id,
            "seed": seed,
            "method": method,
            "stored": {metric: float(record["metrics"][metric]) for metric in deltas},
            "observed": {metric: float(observed[metric]) for metric in deltas},
            "absolute_delta": deltas,
            "tolerance": float(tolerance),
            "source_checkpoint_hash_matches": source_hash_matches,
            "checkpoint_hash_matches": checkpoint_hash_matches,
            "passed": (
                all(delta <= tolerance for delta in deltas.values())
                and source_hash_matches
                and checkpoint_hash_matches
            ),
        }

    def run_time_matched(self, pair_id: str, seed: int, method: str, force: bool = False) -> dict[str, Any]:
        if method not in TIME_MATCHED_METHODS:
            raise ValueError(f"Unknown time-matched method: {method}")
        result_path = self.time_matched_result_path(pair_id, seed, method)
        if result_path.exists() and not force:
            return json.loads(result_path.read_text(encoding="utf-8"))
        pad_record = self.run_method(pair_id, seed, "PAD-TF")
        direct_record = self.run_method(pair_id, seed, "Direct")
        time_cap = float(pad_record["timing"]["total_target"])
        set_seed(seed)
        pair = self.config.pair(pair_id)
        source = pair["source"]
        target = pair["target"]
        source_checkpoint = self.source_checkpoint(source, seed)
        data = self.data(target)
        batch_size = batch_size_for(self.config, target)
        val_loader = make_loader(data.val, batch_size, False, seed, int(self.data_settings["num_workers"]))
        test_loader = make_loader(data.test, batch_size, False, seed, int(self.data_settings["num_workers"]))
        model, transfer = self._target_model(target, source_checkpoint)
        selection_seconds = 0.0
        selection_metadata: dict[str, Any] = {}
        indices = None
        ratio = 1.0
        if method != "FlashST-Full":
            ratio = float(self.training["selection_ratio"])
            selection = select_windows(
                method,
                model,
                data.train,
                data.scaler,
                ratio,
                self.device,
                seed,
                self.config.section("selectors"),
            )
            indices = selection.indices
            selection_seconds = selection.elapsed_seconds
            selection_metadata = selection.metadata
        adaptation_cap = max(0.0, time_cap - selection_seconds)
        train_loader = make_loader(
            data.train,
            batch_size,
            True,
            seed,
            int(self.data_settings["num_workers"]),
            indices,
        )
        checkpoint = (
            self.config.path_for("checkpoints")
            / "analysis"
            / "time_matched"
            / _method_slug(method)
            / pair_id
            / f"seed_{seed}.pt"
        )
        logger = self.logger(pair_id, seed, f"time_matched_{method}")
        fit = fit_model_time_budget(
            model,
            train_loader,
            val_loader,
            data.scaler,
            self.device,
            adaptation_cap,
            int(self.training["full_epochs"]),
            float(self.training["learning_rate"]),
            int(self.training["effective_batch_size"]),
            float(self.training["ssl_weight"]),
            float(self.training["grad_clip"]),
            checkpoint,
            logger,
        )
        metrics = evaluate_model(model, test_loader, data.scaler, self.device)
        total = selection_seconds + fit.elapsed_seconds
        record = self._base_record(pair_id, seed, f"{method}[time-matched]", source_checkpoint, target)
        record.update(
            {
                "base_method": method,
                "selected_strategy": "Direct" if fit.epochs_run == 0 else "Fine-tune",
                "ratio": ratio,
                "epochs": fit.epochs_run,
                "best_epoch": fit.best_epoch,
                "metrics": metrics,
                "timing": {
                    "cap": time_cap,
                    "selection": selection_seconds,
                    "adaptation_cap": adaptation_cap,
                    "adaptation": fit.elapsed_seconds,
                    "total_target": total,
                    "utilization": total / max(time_cap, 1.0e-8),
                },
                "matched_to": {
                    "method": "PAD-TF",
                    "total_target": time_cap,
                    "rmse": float(pad_record["metrics"]["rmse"]),
                },
                "direct_rmse": float(direct_record["metrics"]["rmse"]),
                "selection": selection_metadata,
                "transfer": transfer,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "protocol": "per-pair-and-seed PAD-TF wall-clock cap including selector overhead",
            }
        )
        atomic_write_json(result_path, record)
        logger.info(
            "time_matched_complete pair=%s method=%s seed=%d epochs=%d rmse=%.6f total=%.3f cap=%.3f utilization=%.4f",
            pair_id,
            method,
            seed,
            fit.epochs_run,
            float(metrics["rmse"]),
            total,
            time_cap,
            total / max(time_cap, 1.0e-8),
        )
        return record

    def run_policy(self, pair_id: str, seed: int, policy: str, force: bool = False) -> dict[str, Any]:
        if policy not in ANALYSIS_POLICIES:
            raise ValueError(f"Unknown analysis policy: {policy}")
        result_path = self.policy_result_path(pair_id, seed, policy)
        if result_path.exists() and not force:
            return json.loads(result_path.read_text(encoding="utf-8"))
        pad_record = self.run_method(pair_id, seed, "PAD-TF")
        direct_record = self.run_method(pair_id, seed, "Direct")
        retrain_record = self.run_method(pair_id, seed, "Retrain")
        if pad_record["decision"]["calibration"]["source"] != "lopo":
            raise RuntimeError("Analysis policies require fitted LOPO calibration and a rerun PAD-TF record")
        pair = self.config.pair(pair_id)
        source = pair["source"]
        target = pair["target"]
        source_checkpoint = self.source_checkpoint(source, seed)
        data = self.data(target)
        batch_size = batch_size_for(self.config, target)
        test_loader = make_loader(data.test, batch_size, False, seed, int(self.data_settings["num_workers"]))
        val_loader = make_loader(data.val, batch_size, False, seed, int(self.data_settings["num_workers"]))
        signals = DomainSignals(**pad_record["signals"])
        settings = self.config.section("padtf")
        decision_arguments: dict[str, Any] = {}
        if policy == "dynamic_only":
            decision_arguments["signal_mode"] = "dynamic"
        elif policy == "static_only":
            decision_arguments["signal_mode"] = "static"
            decision_arguments["fixed_epoch_rate"] = float(settings["fixed_response_rate"])
        elif policy == "fixed_b":
            decision_arguments["fixed_epoch_rate"] = float(settings["fixed_response_rate"])
        elif policy == "no_cost":
            decision_arguments["cost_weight"] = 0.0
        elif policy.startswith("lambda_"):
            decision_arguments["cost_weight"] = float(policy.split("_", 1)[1])
        retrain_seconds = float(retrain_record["timing"]["total_target"])
        full_epoch_seconds = retrain_seconds / max(1, int(retrain_record["epochs"]))
        ratio_coverage_record = pad_record.get("ratio_coverage", {})
        ratio_coverage = {
            float(ratio): float(value)
            for ratio, value in ratio_coverage_record.get("values", {}).items()
        }
        selector_estimate = (
            float(pad_record["timing"].get("coverage_profile", 0.0))
            + 0.01 * retrain_seconds
        )
        decision = choose_strategy(
            self.config,
            pair_id,
            signals,
            full_epoch_seconds,
            retrain_seconds,
            selector_estimate,
            ratio_coverage=ratio_coverage or None,
            signed_log_nodes=float(pad_record["epoch_prediction"]["features"]["signed_log_nodes"]),
            flow_target=bool(pad_record["epoch_prediction"]["features"]["flow_target"]),
            **decision_arguments,
        )
        if policy == "static_only":
            policy_overhead = float(pad_record["timing"].get("static", 0.0))
        else:
            policy_overhead = float(pad_record["timing"].get("probe", 0.0))
        logger = self.logger(pair_id, seed, f"policy_{policy}")
        selection_seconds = 0.0
        adaptation_seconds = 0.0
        selection_metadata: dict[str, Any] = {}
        checkpoint: Path | None = None
        reused_main = (
            decision.strategy == pad_record["selected_strategy"]
            and abs(decision.ratio - float(pad_record["ratio"])) <= 1.0e-12
            and decision.epochs == int(pad_record["epochs"])
        )
        set_seed(seed)
        if reused_main:
            metrics = pad_record["metrics"]
            selection_seconds = float(pad_record["timing"].get("selection", 0.0))
            adaptation_seconds = float(pad_record["timing"].get("adaptation", 0.0))
            selection_metadata = dict(pad_record.get("selection", {}))
            if pad_record.get("checkpoint"):
                checkpoint = Path(pad_record["checkpoint"])
        elif decision.strategy == "Direct":
            model, _ = self._target_model(target, source_checkpoint)
            metrics = evaluate_model(model, test_loader, data.scaler, self.device)
        elif decision.strategy == "Retrain":
            metrics = retrain_record["metrics"]
            adaptation_seconds = retrain_seconds
            checkpoint = Path(retrain_record["checkpoint"])
        else:
            model, _ = self._target_model(target, source_checkpoint)
            selection = select_windows(
                "CoreSet-FT",
                model,
                data.train,
                data.scaler,
                decision.ratio,
                self.device,
                seed,
                self.config.section("selectors"),
            )
            selection_seconds = selection.elapsed_seconds
            selection_metadata = selection.metadata
            train_loader = make_loader(
                data.train,
                batch_size,
                True,
                seed,
                int(self.data_settings["num_workers"]),
                selection.indices,
            )
            checkpoint = (
                self.config.path_for("checkpoints")
                / "analysis"
                / policy
                / pair_id
                / f"seed_{seed}.pt"
            )
            fit = fit_model(
                model,
                train_loader,
                val_loader,
                data.scaler,
                self.device,
                decision.epochs,
                float(self.training["learning_rate"]),
                int(self.training["effective_batch_size"]),
                float(self.training["ssl_weight"]),
                float(self.training["grad_clip"]),
                None,
                checkpoint,
                logger,
            )
            adaptation_seconds = fit.elapsed_seconds
            metrics = evaluate_model(model, test_loader, data.scaler, self.device)
        total = policy_overhead + selection_seconds + adaptation_seconds
        record = self._base_record(pair_id, seed, f"PAD-TF[{policy}]", source_checkpoint, target)
        record.update(
            {
                "policy": policy,
                "selected_strategy": decision.strategy,
                "ratio": decision.ratio,
                "epochs": decision.epochs,
                "metrics": metrics,
                "timing": {
                    "policy_overhead": policy_overhead,
                    "selection": selection_seconds,
                    "adaptation": adaptation_seconds,
                    "total_target": total,
                },
                "signals": pad_record["signals"],
                "decision": decision.to_dict(),
                "decision_arguments": decision_arguments,
                "selection": selection_metadata,
                "reused_main_padtf": reused_main,
            }
        )
        if checkpoint is not None and checkpoint.exists():
            record["checkpoint"] = str(checkpoint)
            record["checkpoint_sha256"] = sha256_file(checkpoint)
        atomic_write_json(result_path, record)
        logger.info(
            "policy_complete policy=%s pair=%s seed=%d strategy=%s rho=%.2f epochs=%d rmse=%.6f total=%.3f",
            policy,
            pair_id,
            seed,
            decision.strategy,
            decision.ratio,
            decision.epochs,
            float(metrics["rmse"]),
            total,
        )
        return record

    def run_control(self, control: str, seed: int, force: bool = False) -> dict[str, Any]:
        definitions = {
            "zero_shift": {
                "source": "PEMS04",
                "target": "PEMS04",
                "retrain_pair": "P4",
                "map_pretrain_norm": True,
                "prompt_sign": 1.0,
            },
            "negative_transfer": {
                "source": "PEMS04",
                "target": "PEMS08",
                "retrain_pair": "P3",
                "map_pretrain_norm": False,
                "prompt_sign": -1.0,
            },
        }
        if control not in definitions:
            raise ValueError(f"Unknown strategy control: {control}")
        result_path = self.control_result_path(control, seed)
        if result_path.exists() and not force:
            return json.loads(result_path.read_text(encoding="utf-8"))
        spec = definitions[control]
        source = str(spec["source"])
        target = str(spec["target"])
        map_pretrain_norm = bool(spec["map_pretrain_norm"])
        prompt_sign = float(spec["prompt_sign"])
        source_checkpoint = self.train_source(source, seed)
        retrain = self.run_method(str(spec["retrain_pair"]), seed, "Retrain")
        data = self.data(target)
        source_data = self.data(source)
        batch_size = batch_size_for(self.config, target)
        val_loader = make_loader(data.val, batch_size, False, seed, int(self.data_settings["num_workers"]))
        test_loader = make_loader(data.test, batch_size, False, seed, int(self.data_settings["num_workers"]))
        logger = self.logger(f"control_{control}", seed, "control")

        set_seed(seed)
        direct_model, direct_transfer = self._target_model(
            target,
            source_checkpoint,
            map_pretrain_norm,
            prompt_sign,
        )
        direct_metrics = evaluate_model(direct_model, test_loader, data.scaler, self.device)

        set_seed(seed)
        full_model, full_transfer = self._target_model(
            target,
            source_checkpoint,
            map_pretrain_norm,
            prompt_sign,
        )
        full_loader = make_loader(
            data.train,
            batch_size,
            True,
            seed,
            int(self.data_settings["num_workers"]),
        )
        control_checkpoint_dir = self.config.path_for("checkpoints") / "analysis" / "controls" / control / f"seed_{seed}"
        full_checkpoint = control_checkpoint_dir / "full.pt"
        full_fit = fit_model(
            full_model,
            full_loader,
            val_loader,
            data.scaler,
            self.device,
            int(self.training["full_epochs"]),
            float(self.training["learning_rate"]),
            int(self.training["effective_batch_size"]),
            float(self.training["ssl_weight"]),
            float(self.training["grad_clip"]),
            None,
            full_checkpoint,
            logger,
        )
        full_metrics = evaluate_model(full_model, test_loader, data.scaler, self.device)

        settings = self.config.section("padtf")
        with elapsed_timer(self.device) as static_timer:
            static = static_discrepancy(
                source_data,
                data,
                self.graph(source),
                self.graph(target),
                float(settings["epsilon"]),
            )
        probe_gap = max(
            int(settings.get("probe_block_gap", 0)),
            int(self.data_settings["history"]) + int(self.data_settings["horizon"]),
        )
        probe_train, probe_val = blocked_stratified_probe_split(
            data.train,
            float(settings["probe_ratio"]),
            float(settings["probe_train_fraction"]),
            seed,
            probe_gap,
        )
        probe_train_loader = make_loader(data.train, batch_size, True, seed, indices=probe_train)
        probe_val_loader = make_loader(data.train, batch_size, False, seed, indices=probe_val)
        set_seed(seed)
        probe_model, probe_transfer = self._target_model(
            target,
            source_checkpoint,
            map_pretrain_norm,
            prompt_sign,
        )
        with elapsed_timer(self.device) as zero_timer:
            zero_shot_mae = validation_mae(probe_model, probe_val_loader, data.scaler, self.device)
            naive_mae = persistence_mae(probe_val_loader, data.scaler, self.device)
        probe_checkpoint = control_checkpoint_dir / "probe.pt"
        probe_fit = fit_model(
            probe_model,
            probe_train_loader,
            probe_val_loader,
            data.scaler,
            self.device,
            int(settings["probe_epochs"]),
            float(self.training["learning_rate"]),
            int(self.training["effective_batch_size"]),
            float(self.training["ssl_weight"]),
            float(self.training["grad_clip"]),
            None,
            probe_checkpoint,
            logger,
        )
        final_probe_mae = probe_fit.best_val_mae
        probe_gain = (zero_shot_mae - final_probe_mae) / (zero_shot_mae + float(settings["epsilon"]))
        signals = combine_signals(
            static,
            zero_shot_mae,
            naive_mae,
            probe_gain,
            float(settings["alpha"]),
            float(settings["epsilon"]),
        )
        retrain_seconds = float(retrain["timing"]["total_target"])
        full_epoch_seconds = retrain_seconds / max(1, int(retrain["epochs"]))
        with elapsed_timer(self.device) as grid_timer:
            decision = choose_strategy(
                self.config,
                control,
                signals,
                full_epoch_seconds,
                retrain_seconds,
                0.01 * retrain_seconds,
            )
        selection_seconds = 0.0
        adaptation_seconds = 0.0
        selection_metadata: dict[str, Any] = {}
        pad_checkpoint: Path | None = None
        set_seed(seed)
        if decision.strategy == "Direct":
            pad_metrics = direct_metrics
        elif decision.strategy == "Retrain":
            pad_metrics = retrain["metrics"]
            adaptation_seconds = retrain_seconds
            pad_checkpoint = Path(retrain["checkpoint"])
        else:
            pad_model, _ = self._target_model(
                target,
                source_checkpoint,
                map_pretrain_norm,
                prompt_sign,
            )
            selection = select_windows(
                "CoreSet-FT",
                pad_model,
                data.train,
                data.scaler,
                decision.ratio,
                self.device,
                seed,
                self.config.section("selectors"),
            )
            selection_seconds = selection.elapsed_seconds
            selection_metadata = selection.metadata
            train_loader = make_loader(data.train, batch_size, True, seed, indices=selection.indices)
            pad_checkpoint = control_checkpoint_dir / "pad_tf.pt"
            pad_fit = fit_model(
                pad_model,
                train_loader,
                val_loader,
                data.scaler,
                self.device,
                decision.epochs,
                float(self.training["learning_rate"]),
                int(self.training["effective_batch_size"]),
                float(self.training["ssl_weight"]),
                float(self.training["grad_clip"]),
                None,
                pad_checkpoint,
                logger,
            )
            adaptation_seconds = pad_fit.elapsed_seconds
            pad_metrics = evaluate_model(pad_model, test_loader, data.scaler, self.device)
        probe_seconds = static_timer["seconds"] + zero_timer["seconds"] + probe_fit.elapsed_seconds + grid_timer["seconds"]
        pad_total = probe_seconds + selection_seconds + adaptation_seconds
        weight = float(settings["lambda"])
        direct_rmse = float(direct_metrics["rmse"])

        def utility(metrics: dict[str, Any], seconds: float) -> float:
            return deployment_utility(
                direct_rmse,
                float(metrics["rmse"]),
                seconds,
                retrain_seconds,
                weight,
            )

        executed = [
            ("Direct", direct_metrics, 0.0),
            ("Fine-tune", full_metrics, full_fit.elapsed_seconds),
            ("Retrain", retrain["metrics"], retrain_seconds),
        ]
        oracle_family, oracle_metrics, oracle_seconds = max(
            executed,
            key=lambda item: utility(item[1], item[2]),
        )
        record = {
            "control": control,
            "seed": seed,
            "source": source,
            "target": target,
            "source_checkpoint": str(source_checkpoint),
            "source_checkpoint_sha256": sha256_file(source_checkpoint),
            "config_hash": self.config.hash,
            "upstream_commit": self.config.section("project")["upstream_commit"],
            "transfer_control": {
                "map_pretrain_norm": map_pretrain_norm,
                "prompt_sign": prompt_sign,
                "direct": direct_transfer,
                "full": full_transfer,
                "probe": probe_transfer,
            },
            "executed": {
                "Direct": {"metrics": direct_metrics, "total_target": 0.0, "utility": utility(direct_metrics, 0.0)},
                "FlashST-Full": {
                    "metrics": full_metrics,
                    "total_target": full_fit.elapsed_seconds,
                    "utility": utility(full_metrics, full_fit.elapsed_seconds),
                },
                "Retrain": {
                    "metrics": retrain["metrics"],
                    "total_target": retrain_seconds,
                    "utility": utility(retrain["metrics"], retrain_seconds),
                },
            },
            "oracle": {
                "family": oracle_family,
                "metrics": oracle_metrics,
                "total_target": oracle_seconds,
                "utility": utility(oracle_metrics, oracle_seconds),
            },
            "pad_tf": {
                "selected_strategy": decision.strategy,
                "ratio": decision.ratio,
                "epochs": decision.epochs,
                "metrics": pad_metrics,
                "timing": {
                    "probe": probe_seconds,
                    "selection": selection_seconds,
                    "adaptation": adaptation_seconds,
                    "total_target": pad_total,
                },
                "utility": utility(pad_metrics, pad_total),
                "signals": signals.__dict__,
                "decision": decision.to_dict(),
                "selection": selection_metadata,
            },
            "family_correct": decision.strategy == oracle_family,
        }
        if pad_checkpoint is not None and pad_checkpoint.exists():
            record["pad_tf"]["checkpoint"] = str(pad_checkpoint)
            record["pad_tf"]["checkpoint_sha256"] = sha256_file(pad_checkpoint)
        atomic_write_json(result_path, record)
        logger.info(
            "control_complete control=%s seed=%d oracle=%s pad=%s correct=%s",
            control,
            seed,
            oracle_family,
            decision.strategy,
            record["family_correct"],
        )
        return record

    def run_calibration_grid(self, pair_id: str, seed: int, force: bool = False) -> list[dict[str, Any]]:
        set_seed(seed)
        direct = self.run_method(pair_id, seed, "Direct")
        retrain = self.run_method(pair_id, seed, "Retrain")
        pad_record = self.run_method(pair_id, seed, "PAD-TF")
        pair = self.config.pair(pair_id)
        target = pair["target"]
        source_checkpoint = self.source_checkpoint(pair["source"], seed)
        data = self.data(target)
        batch_size = batch_size_for(self.config, target)
        val_loader = make_loader(data.val, batch_size, False, seed)
        test_loader = make_loader(data.test, batch_size, False, seed)
        settings = self.config.section("padtf")
        output_dir = self.config.path_for("results") / "calibration" / "grid"
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = self.config.path_for("checkpoints") / "calibration" / pair_id / f"seed_{seed}"
        direct_mae = float(direct["metrics"]["mae"])
        retrain_gain = max(0.0, (direct_mae - float(retrain["metrics"]["mae"])) / max(direct_mae, 1.0e-8))
        phi = max(float(pad_record["signals"]["phi"]), 1.0e-8)
        retrain_fraction = float(min(1.0, retrain_gain / phi))
        records = []
        for ratio in settings["ratios"]:
            ratio = float(ratio)
            requested_epochs = [int(epoch) for epoch in settings["epochs"]]
            result_paths = {
                epochs: output_dir / f"{pair_id}_seed{seed}_rho{ratio:.2f}_e{epochs:02d}.json"
                for epochs in requested_epochs
            }
            existing = {
                epochs: json.loads(path.read_text(encoding="utf-8"))
                for epochs, path in result_paths.items()
                if path.exists() and not force
            }
            if len(existing) == len(requested_epochs):
                records.extend(existing[epochs] for epochs in requested_epochs)
                continue
            set_seed(seed)
            selector_model, _ = self._target_model(target, source_checkpoint)
            selection = select_windows(
                "CoreSet-FT",
                selector_model,
                data.train,
                data.scaler,
                ratio,
                self.device,
                seed,
                self.config.section("selectors"),
            )
            set_seed(seed)
            model, transfer = self._target_model(target, source_checkpoint)
            train_loader = make_loader(data.train, batch_size, True, seed, indices=selection.indices)
            checkpoints = {
                epochs: checkpoint_dir / f"rho{ratio:.2f}_e{epochs:02d}.pt"
                for epochs in requested_epochs
            }
            logger = self.logger(pair_id, seed, f"grid_r{ratio:.2f}")
            fits = fit_model_milestones(
                model,
                train_loader,
                val_loader,
                data.scaler,
                self.device,
                requested_epochs,
                float(self.training["learning_rate"]),
                int(self.training["effective_batch_size"]),
                float(self.training["ssl_weight"]),
                float(self.training["grad_clip"]),
                checkpoints,
                logger,
            )
            for epochs in requested_epochs:
                result_path = result_paths[epochs]
                checkpoint = checkpoints[epochs]
                self._load_trained_model(model, checkpoint)
                metrics = evaluate_model(model, test_loader, data.scaler, self.device)
                relative_gain = max(0.0, (direct_mae - float(metrics["mae"])) / max(direct_mae, 1.0e-8))
                record = {
                    "pair": pair_id,
                    "seed": seed,
                    "source": pair["source"],
                    "target": target,
                    "ratio": ratio,
                    "epochs": epochs,
                    "probe_gain": float(pad_record["signals"]["probe_gain"]),
                    "phi": phi,
                    "direct_mae": direct_mae,
                    "mae": float(metrics["mae"]),
                    "rmse": float(metrics["rmse"]),
                    "captured_fraction": float(min(1.0, relative_gain / phi)),
                    "retrain_captured_fraction": retrain_fraction,
                    "selection_seconds": selection.elapsed_seconds,
                    "adaptation_seconds": fits[epochs].elapsed_seconds,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "source_checkpoint_sha256": sha256_file(source_checkpoint),
                    "config_hash": self.config.hash,
                    "transfer": transfer,
                    "training_protocol": "single_trajectory_milestone",
                }
                atomic_write_json(result_path, record)
                records.append(record)
        return records

    def doctor(self) -> dict[str, Any]:
        upstream = self.config.path_for("upstream")
        expected_commit = self.config.section("project")["upstream_commit"]
        actual_commit = command_output(["git", "rev-parse", "HEAD"], cwd=upstream) if upstream.exists() else "missing"
        datasets = {}
        for name in self.config.raw["datasets"]:
            spec = self.config.dataset(name)
            values_ok = Path(spec["values_path"]).exists()
            graph_ok = Path(spec["graph_path"]).exists()
            shape = None
            graph_shape = None
            graph_nonzero = None
            graph_density = None
            if values_ok:
                values = self.data(name).values
                shape = list(values.shape)
            if graph_ok:
                adjacency = load_adjacency(self.config, name)
                graph_shape = list(adjacency.shape)
                graph_nonzero = int((adjacency != 0).sum())
                graph_density = graph_nonzero / max(1, adjacency.size)
            datasets[name] = {
                "values": str(spec["values_path"]),
                "values_ok": values_ok,
                "shape": shape,
                "graph": str(spec["graph_path"]),
                "graph_ok": graph_ok,
                "graph_shape": graph_shape,
                "graph_nonzero": graph_nonzero,
                "graph_density": graph_density,
            }
        return {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "upstream": str(upstream),
            "upstream_expected_commit": expected_commit,
            "upstream_actual_commit": actual_commit,
            "upstream_ok": actual_commit == expected_commit,
            "config_hash": self.config.hash,
            "datasets": datasets,
        }

    def smoke(self, pair_id: str, seed: int) -> dict[str, Any]:
        set_seed(seed)
        pair = self.config.pair(pair_id)
        source = pair["source"]
        target = pair["target"]
        source_data = self.data(source)
        target_data = self.data(target)
        source_model = FlashSTTransferModel(self.config, source, self.graph(source), self.device, "source").to(self.device)
        source_batch = next(iter(make_loader(source_data.train, 2, False, seed)))
        source_model.train()
        optimizer = torch.optim.Adam(source_model.parameters(), lr=float(self.training["learning_rate"]))
        inputs, targets = (tensor.to(self.device) for tensor in source_batch)
        prediction, representation = source_model(inputs)
        loss = torch.abs(source_data.scaler.inverse_tensor(prediction) - source_data.scaler.inverse_tensor(targets)).mean()
        if representation is not None:
            loss = loss + 0.01 * torch.square(representation).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        temporary = self.config.path_for("results") / "smoke" / f"{pair_id}_{seed}_source.pt"
        temporary.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": source_model.state_dict()}, temporary)
        target_model, transfer = self._target_model(target, temporary)
        target_batch = next(iter(make_loader(target_data.train, 2, False, seed)))
        target_inputs = target_batch[0].to(self.device)
        target_prediction, _ = target_model(target_inputs)
        selector_dataset = Subset(target_data.train, range(min(64, len(target_data.train))))
        selector_results = {}
        for method in ("Random-FT", "CoreSet-FT", "Proxy-FT", "ELFS-FT", "BADGE-FT", "UHerding-FT"):
            selection = select_windows(
                method,
                target_model,
                selector_dataset,
                target_data.scaler,
                0.25,
                self.device,
                seed,
                self.config.section("selectors"),
            )
            selector_results[method] = {
                "selected": len(selection.indices),
                "elapsed_seconds": selection.elapsed_seconds,
                "metadata": selection.metadata,
            }
        set_seed(seed)
        milestone_model, _ = self._target_model(target, temporary)
        milestone_train = Subset(target_data.train, range(min(64, len(target_data.train))))
        milestone_val = Subset(target_data.val, range(min(32, len(target_data.val))))
        milestone_root = self.config.path_for("results") / "smoke" / "milestones"
        milestone_paths = {epoch: milestone_root / f"{pair_id}_{seed}_e{epoch}.pt" for epoch in (1, 2)}
        milestone_fits = fit_model_milestones(
            milestone_model,
            make_loader(milestone_train, 8, True, seed),
            make_loader(milestone_val, 8, False, seed),
            target_data.scaler,
            self.device,
            (1, 2),
            float(self.training["learning_rate"]),
            16,
            float(self.training["ssl_weight"]),
            float(self.training["grad_clip"]),
            milestone_paths,
            self.logger(pair_id, seed, "smoke_milestones"),
        )
        result = {
            "pair": pair_id,
            "seed": seed,
            "source_shape": list(prediction.shape),
            "target_shape": list(target_prediction.shape),
            "source_loss": float(loss.item()),
            "transfer": transfer,
            "source_trainable": trainable_parameter_count(source_model),
            "target_trainable": trainable_parameter_count(target_model),
            "selectors": selector_results,
            "milestones": {
                str(epoch): {
                    "elapsed_seconds": fit.elapsed_seconds,
                    "val_mae": fit.best_val_mae,
                    "checkpoint": str(milestone_paths[epoch]),
                }
                for epoch, fit in milestone_fits.items()
            },
        }
        atomic_write_json(self.config.path_for("results") / "smoke" / f"{pair_id}_{seed}.json", result)
        return result
