"""Privacy-preserving local review feedback.

Only category-level metadata is stored. Document text, page images, coordinates,
and secret previews are deliberately excluded.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable

DATA_DIR = Path(__file__).resolve().parent / "data"
FEEDBACK_PATH = DATA_DIR / "redaction_feedback.jsonl"


def record_review_metadata(
    *,
    filename: str,
    custom_prompt_used: bool,
    rows: Iterable[dict[str, object]],
) -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    count = 0
    with FEEDBACK_PATH.open("a", encoding="utf-8") as handle:
        for row in rows:
            payload = {
                "timestamp": timestamp,
                "file_extension": Path(filename).suffix.lower(),
                "custom_prompt_used": bool(custom_prompt_used),
                "category": str(row.get("category", "unknown")),
                "accepted": bool(row.get("accepted", False)),
                "confidence": round(float(row.get("confidence", 0.0)), 4),
                "source": str(row.get("source", "unknown")),
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1
    return count


def category_confidence_adjustments(minimum_samples: int = 6) -> dict[str, float]:
    """Return small category-level calibration adjustments from prior reviews.

    The adjustment is deliberately capped at +/-0.08 so user feedback cannot
    overwhelm the underlying detectors.
    """

    if not FEEDBACK_PATH.exists():
        return {}

    totals: dict[str, int] = defaultdict(int)
    accepted: dict[str, int] = defaultdict(int)
    try:
        with FEEDBACK_PATH.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                    category = str(item.get("category", ""))
                    if not category:
                        continue
                    totals[category] += 1
                    accepted[category] += int(bool(item.get("accepted", False)))
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
    except OSError:
        return {}

    adjustments: dict[str, float] = {}
    for category, total in totals.items():
        if total < minimum_samples:
            continue
        rate = accepted[category] / total
        adjustments[category] = max(-0.08, min(0.08, (rate - 0.5) * 0.16))
    return adjustments


def apply_local_calibration(suggestions: list[object]) -> None:
    adjustments = category_confidence_adjustments()
    for suggestion in suggestions:
        category = getattr(suggestion, "category", "")
        adjustment = adjustments.get(category, 0.0)
        if adjustment:
            confidence = float(getattr(suggestion, "confidence", 0.0))
            setattr(suggestion, "confidence", max(0.0, min(0.995, confidence + adjustment)))
