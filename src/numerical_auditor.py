"""Conservative numerical comparison for matching metrics and units."""

import re
from typing import List, Tuple

from .schemas import Claim, Evidence

_NUMBER_REGEX = re.compile(
    r"(?<![\w.])[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\w.])"
)
_UNIT_REGEX = re.compile(
    r"^\s*(%|percent\b|milliseconds?\b|ms\b|seconds?\b|secs?\b|s\b|"
    r"minutes?\b|mins?\b|hours?\b|hrs?\b|GB\b|MB\b|KB\b)",
    re.IGNORECASE,
)
_METRIC_REGEX = re.compile(
    r"\b(accuracy|precision|recall|f1(?:[- ]score)?|error(?: rate)?|loss|"
    r"latency|runtime|training time|inference time|throughput|score)\b",
    re.IGNORECASE,
)


def _measurement(text: str) -> Tuple[str, float, str] | None:
    metrics = list(_METRIC_REGEX.finditer(text))
    numbers = list(_NUMBER_REGEX.finditer(text))
    if not metrics or not numbers:
        return None

    metric = min(
        metrics,
        key=lambda match: min(
            abs(match.start() - number.end()) for number in numbers
        ),
    )
    number = min(
        numbers,
        key=lambda match: abs(match.start() - metric.end())
        if match.start() >= metric.end()
        else abs(metric.start() - match.end()),
    )
    if min(abs(metric.start() - number.end()), abs(number.start() - metric.end())) > 40:
        return None

    unit_match = _UNIT_REGEX.match(text[number.end() :])
    unit = unit_match.group(1).lower() if unit_match else "unspecified"
    if unit in {"percent", "%"}:
        unit = "%"
    elif unit in {"ms", "millisecond", "milliseconds"}:
        unit = "milliseconds"
    elif unit in {"s", "sec", "secs", "second", "seconds"}:
        unit = "seconds"
    value = float(number.group().replace(",", ""))
    return metric.group().lower(), value, unit

def check_numerical_consistency(claims: List[Claim], evidences: List[Evidence]) -> List[Tuple[int, str]]:
    """Return review prompts for comparable metric-and-unit pairs that differ.

    A difference is not automatically an error: it may reflect different settings or
    experimental conditions, which this lightweight check cannot establish.
    """
    ev_by_claim = {}
    for ev in evidences:
        ev_by_claim.setdefault(ev.claim_id, []).append(ev)
    warnings: List[Tuple[int, str]] = []
    for claim in claims:
        claim_measurement = _measurement(claim.text)
        if claim_measurement is None:
            continue
        for ev in ev_by_claim.get(claim.claim_id, []):
            evidence_measurement = _measurement(ev.text)
            if evidence_measurement is None:
                continue
            metric, claim_value, unit = claim_measurement
            evidence_metric, evidence_value, evidence_unit = evidence_measurement
            if metric != evidence_metric or unit != evidence_unit:
                continue
            if claim_value == evidence_value:
                continue

            relative_difference = (
                abs(claim_value - evidence_value) / abs(claim_value)
                if claim_value
                else float("inf")
            )
            if relative_difference > 0.1:
                warnings.append(
                    (
                        claim.claim_id,
                        f"Review: {metric} differs ({claim_value:g} {unit} vs "
                        f"{evidence_value:g} {unit}; relative difference "
                        f"{relative_difference:.0%}). This is not automatically an error; "
                        "verify the experimental context.",
                    )
                )
                break
    return warnings
