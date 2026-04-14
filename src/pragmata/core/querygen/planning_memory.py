"""Stage-1 planning memory helpers for synthetic query generation."""

import hashlib
import json

from pragmata.core.schemas.querygen_input import QueryGenSpec


def _serialize_spec_content(
    spec: QueryGenSpec,
) -> str:
    """Build a deterministic content-only serialization for a querygen spec.

    Args:
        spec: Resolved query-generation specification.

    Returns:
        A canonical JSON string suitable for stable hashing.
    """
    payload = spec.model_dump(mode="json")

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def fingerprint_querygen_spec(spec: QueryGenSpec) -> str:
    """Return a deterministic SHA-256 fingerprint for a querygen spec.

    Args:
        spec: Resolved query-generation specification.

    Returns:
        Stable SHA-256 hex digest of the canonical serialized spec content.
    """
    serialized = _serialize_spec_content(spec)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
