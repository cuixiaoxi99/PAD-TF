from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from .config import ExperimentConfig
from .graph import GraphBundle


def _load_upstream_classes(upstream_root: Path) -> tuple[type[nn.Module], type[nn.Module]]:
    if not (upstream_root / "model" / "PromptNet.py").exists():
        raise FileNotFoundError(f"FlashST checkout not found at {upstream_root}")
    root_text = str(upstream_root)
    model_text = str(upstream_root / "model")
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    if model_text not in sys.path:
        sys.path.insert(0, model_text)
    from model.PromptNet import PromptNet
    from model.STGCN.stgcn import STGCN

    return PromptNet, STGCN


def _base_arguments(
    config: ExperimentConfig,
    dataset_name: str,
    graph: GraphBundle,
    device: torch.device,
    mode: str,
) -> SimpleNamespace:
    model_config = config.section("model")
    data_config = config.section("data")
    dataset = config.dataset(dataset_name)
    return SimpleNamespace(
        mode=mode,
        model="STGCN",
        num_nodes=int(dataset["nodes"]),
        input_base_dim=1,
        input_extra_dim=2,
        output_dim=1,
        his=int(data_config["history"]),
        pred=int(data_config["horizon"]),
        embed_dim=int(model_config["embed_dim"]),
        node_dim=int(model_config["node_dim"]),
        num_layer=int(model_config["num_layers"]),
        temp_dim_tid=int(model_config["temporal_dim"]),
        temp_dim_diw=int(model_config["temporal_dim"]),
        if_time_in_day=True,
        if_day_in_week=True,
        if_spatial=True,
        data_type=dataset["feature"],
        device=device,
        dataset_use=[dataset_name],
        dataset_test=dataset_name,
        A_dict={dataset_name: torch.from_numpy(graph.normalized_adjacency).to(device)},
        lpls_dict={dataset_name: torch.from_numpy(graph.laplacian_pe).to(device)},
    )


def _stgcn_arguments(config: ExperimentConfig, nodes: int) -> SimpleNamespace:
    model_config = config.section("model")
    return SimpleNamespace(
        Ks=int(model_config["ks"]),
        Kt=int(model_config["kt"]),
        blocks1=list(model_config["blocks"]),
        drop_prob=float(model_config["dropout"]),
        outputl_ks=3,
        num_nodes=nodes,
        input_window=int(config.section("data")["history"]),
        output_window=int(config.section("data")["horizon"]),
    )


class FlashSTTransferModel(nn.Module):
    def __init__(
        self,
        config: ExperimentConfig,
        dataset_name: str,
        graph: GraphBundle,
        device: torch.device,
        stage: str,
    ) -> None:
        super().__init__()
        if stage not in {"source", "target", "retrain"}:
            raise ValueError(f"Unknown model stage: {stage}")
        upstream_root = config.path_for("upstream")
        PromptNet, STGCN = _load_upstream_classes(upstream_root)
        upstream_mode = {"source": "pretrain", "target": "eval", "retrain": "ori"}[stage]
        arguments = _base_arguments(config, dataset_name, graph, device, upstream_mode)
        stgcn_arguments = _stgcn_arguments(config, int(arguments.num_nodes))
        graph_tensors = {dataset_name: torch.from_numpy(graph.chebyshev).to(device)}
        input_dimension = 1 if stage == "retrain" else int(config.section("model")["embed_dim"]) * 4
        self.dataset_name = dataset_name
        self.stage = stage
        self.arguments = arguments
        self.register_buffer("normalized_adjacency", arguments.A_dict[dataset_name], persistent=False)
        self.register_buffer("laplacian_pe", arguments.lpls_dict[dataset_name], persistent=False)
        self.predictor = STGCN(
            stgcn_arguments,
            graph_tensors,
            [dataset_name],
            [dataset_name],
            int(arguments.num_nodes),
            input_dimension,
            1,
            device,
            upstream_mode,
        )
        self.prompt = None if stage == "retrain" else PromptNet(arguments)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.prompt is None:
            return self.predictor(inputs[..., :1], self.dataset_name), None
        representation = self.prompt(
            inputs[..., :1],
            inputs,
            None,
            self.normalized_adjacency,
            self.laplacian_pe,
            True,
        )
        return self.predictor(representation, self.dataset_name), representation

    def configure_target_trainable(self) -> None:
        if self.stage != "target":
            return
        for parameter in self.predictor.parameters():
            parameter.requires_grad = False
        for module in (
            self.predictor.st_conv1.ln_eval,
            self.predictor.st_conv2.ln_eval,
            self.predictor.output.ln_eval,
            self.predictor.output.fc_pretrain,
        ):
            for parameter in module.parameters():
                parameter.requires_grad = True
        if self.prompt is not None:
            for parameter in self.prompt.parameters():
                parameter.requires_grad = True


def transferable_state(
    target: FlashSTTransferModel,
    source_state: dict[str, torch.Tensor],
    map_pretrain_norm: bool = False,
    prompt_sign: float = 1.0,
) -> dict[str, torch.Tensor]:
    target_state = target.state_dict()
    selected: dict[str, torch.Tensor] = {}
    for key, value in source_state.items():
        if ".ln_pretrain." in key:
            if map_pretrain_norm:
                target_key = key.replace(".ln_pretrain.", ".ln_eval.")
                if target_key in target_state and target_state[target_key].shape == value.shape:
                    selected[target_key] = value
            continue
        if ".ln_eval." in key:
            continue
        if key in target_state and target_state[key].shape == value.shape:
            selected[key] = value * prompt_sign if key.startswith("prompt.") else value
    return selected


def load_source_into_target(
    model: FlashSTTransferModel,
    checkpoint: str | Path,
    map_pretrain_norm: bool = False,
    prompt_sign: float = 1.0,
) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu")
    source_state = payload.get("state_dict", payload)
    selected = transferable_state(model, source_state, map_pretrain_norm, prompt_sign)
    missing, unexpected = model.load_state_dict(selected, strict=False)
    model.configure_target_trainable()
    return {
        "loaded_tensors": len(selected),
        "missing_tensors": list(missing),
        "unexpected_tensors": list(unexpected),
        "mapped_pretrain_norm": map_pretrain_norm,
        "prompt_sign": prompt_sign,
    }


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
