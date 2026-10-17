from __future__ import annotations

import argparse
import json
from typing import Any

from .aggregate import aggregate
from .audit import audit_protocol
from .time_matched import aggregate_time_matched
from .calibration import fit_lopo_calibration
from .config import DEFAULT_CONFIG, ExperimentConfig
from .runner import PAPER_METHODS, TIME_MATCHED_METHODS, ExperimentRunner


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PAD-TF FlashST experiment runner")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--device", default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("doctor")

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--pair", default="P3")
    smoke.add_argument("--seed", type=int, default=2024)

    probe = subparsers.add_parser("run-probe")
    probe.add_argument("--pair", required=True)
    probe.add_argument("--seed", type=int, required=True)
    probe.add_argument("--force", action="store_true")

    source = subparsers.add_parser("train-source")
    source.add_argument("--dataset", required=True)
    source.add_argument("--seed", type=int, required=True)
    source.add_argument("--force", action="store_true")

    run = subparsers.add_parser("run")
    run.add_argument("--pair", required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--method", required=True)
    run.add_argument("--force", action="store_true")

    pair = subparsers.add_parser("run-pair")
    pair.add_argument("--pair", required=True)
    pair.add_argument("--seed", type=int, required=True)
    pair.add_argument("--methods", nargs="+", choices=PAPER_METHODS, default=None)
    pair.add_argument("--force", action="store_true")

    verify = subparsers.add_parser("verify-result")
    verify.add_argument("--pair", required=True)
    verify.add_argument("--seed", type=int, required=True)
    verify.add_argument("--method", choices=PAPER_METHODS, required=True)
    verify.add_argument("--tolerance", type=float, default=1.0e-4)

    main = subparsers.add_parser("run-main")
    main.add_argument("--pairs", nargs="*", default=None)
    main.add_argument("--seeds", nargs="*", type=int, default=None)
    main.add_argument("--force", action="store_true")

    grid = subparsers.add_parser("run-grid")
    grid.add_argument("--pair", required=True)
    grid.add_argument("--seed", type=int, default=2024)
    grid.add_argument("--force", action="store_true")

    policy = subparsers.add_parser("run-policy")
    policy.add_argument("--pair", required=True)
    policy.add_argument("--seed", type=int, required=True)
    policy.add_argument("--policy", required=True)
    policy.add_argument("--force", action="store_true")

    control = subparsers.add_parser("run-control")
    control.add_argument("--control", required=True)
    control.add_argument("--seed", type=int, required=True)
    control.add_argument("--force", action="store_true")

    time_matched = subparsers.add_parser("run-time-matched")
    time_matched.add_argument("--pair", required=True)
    time_matched.add_argument("--seed", type=int, required=True)
    time_matched.add_argument("--method", choices=TIME_MATCHED_METHODS, required=True)
    time_matched.add_argument("--force", action="store_true")

    subparsers.add_parser("fit-calibration")
    subparsers.add_parser("aggregate")
    subparsers.add_parser("aggregate-time-matched")
    subparsers.add_parser("audit-protocol")
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    config = ExperimentConfig(arguments.config)
    result: Any
    if arguments.command == "aggregate":
        result = aggregate(config)
    elif arguments.command == "aggregate-time-matched":
        result = aggregate_time_matched(config)
    elif arguments.command == "audit-protocol":
        result = audit_protocol(config)
    elif arguments.command == "fit-calibration":
        result = {"calibration": str(fit_lopo_calibration(config))}
    else:
        runner = ExperimentRunner(config, arguments.device)
        if arguments.command == "doctor":
            result = runner.doctor()
        elif arguments.command == "smoke":
            result = runner.smoke(arguments.pair, arguments.seed)
        elif arguments.command == "run-probe":
            result = runner.run_probe(arguments.pair, arguments.seed, arguments.force)
        elif arguments.command == "train-source":
            result = {"checkpoint": str(runner.train_source(arguments.dataset, arguments.seed, arguments.force))}
        elif arguments.command == "run":
            result = runner.run_method(arguments.pair, arguments.seed, arguments.method, arguments.force)
        elif arguments.command == "run-pair":
            result = runner.run_pair(arguments.pair, arguments.seed, arguments.force, arguments.methods)
        elif arguments.command == "verify-result":
            result = runner.verify_result(
                arguments.pair,
                arguments.seed,
                arguments.method,
                arguments.tolerance,
            )
        elif arguments.command == "run-main":
            pairs = arguments.pairs or list(config.raw["pairs"])
            seeds = arguments.seeds or list(config.section("training")["seeds"])
            result = []
            for seed in seeds:
                for pair_id in pairs:
                    result.extend(runner.run_pair(pair_id, seed, arguments.force))
        elif arguments.command == "run-grid":
            result = runner.run_calibration_grid(arguments.pair, arguments.seed, arguments.force)
        elif arguments.command == "run-policy":
            result = runner.run_policy(arguments.pair, arguments.seed, arguments.policy, arguments.force)
        elif arguments.command == "run-control":
            result = runner.run_control(arguments.control, arguments.seed, arguments.force)
        elif arguments.command == "run-time-matched":
            result = runner.run_time_matched(arguments.pair, arguments.seed, arguments.method, arguments.force)
        else:
            raise RuntimeError(arguments.command)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
