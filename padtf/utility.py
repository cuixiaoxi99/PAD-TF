from __future__ import annotations


def relative_error_reduction(reference_error: float, candidate_error: float, epsilon: float = 1.0e-8) -> float:
    denominator = max(float(reference_error), float(epsilon))
    return (float(reference_error) - float(candidate_error)) / denominator


def normalized_target_cost(total_seconds: float, retrain_seconds: float, clip: bool = False) -> float:
    denominator = max(float(retrain_seconds), 1.0e-8)
    cost = float(total_seconds) / denominator
    return min(cost, 1.0) if clip else cost


def estimated_utility(benefit: float, normalized_cost: float, cost_weight: float) -> float:
    return float(benefit) - float(cost_weight) * float(normalized_cost)


def deployment_utility(
    reference_error: float,
    candidate_error: float,
    total_seconds: float,
    retrain_seconds: float,
    cost_weight: float,
) -> float:
    benefit = relative_error_reduction(reference_error, candidate_error)
    cost = normalized_target_cost(total_seconds, retrain_seconds)
    return estimated_utility(benefit, cost, cost_weight)
