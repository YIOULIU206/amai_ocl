"""Immutable soft-constraint records and JSONL library loading."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from .json_utils import canonical_json, jsonable


class LibraryError(ValueError):
    """Raised when a constraint library is invalid."""


class ConstraintResponse(str, Enum):
    WARN = "warn"
    REVISE = "revise"
    BLOCK = "block"
    ESCALATE = "escalate"


class ConstraintStatus(str, Enum):
    CANDIDATE = "candidate"
    APPROVED = "approved"
    REJECTED = "rejected"


def _nonempty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LibraryError(f"{name} must be a non-empty string")
    return value.strip()


def _string_tuple(value: Iterable[str], name: str, *, allow_empty: bool = True) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise LibraryError(f"{name} must be a sequence of strings")
    result = tuple(_nonempty(item, name) for item in value)
    if not allow_empty and not result:
        raise LibraryError(f"{name} must not be empty")
    return result


@dataclass(frozen=True, slots=True)
class SoftConstraint:
    constraint_id: str
    action_types: tuple[str, ...]
    tactic_type: str
    trigger_pattern: str
    keywords: tuple[str, ...]
    instruction: str
    response: ConstraintResponse
    status: ConstraintStatus
    source_episode_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "constraint_id", _nonempty(self.constraint_id, "constraint_id"))
        object.__setattr__(
            self,
            "action_types",
            _string_tuple(self.action_types, "action_types", allow_empty=False),
        )
        object.__setattr__(self, "tactic_type", _nonempty(self.tactic_type, "tactic_type"))
        object.__setattr__(
            self,
            "trigger_pattern",
            _nonempty(self.trigger_pattern, "trigger_pattern"),
        )
        object.__setattr__(self, "keywords", _string_tuple(self.keywords, "keywords"))
        object.__setattr__(self, "instruction", _nonempty(self.instruction, "instruction"))
        if not isinstance(self.response, ConstraintResponse):
            try:
                object.__setattr__(self, "response", ConstraintResponse(self.response))
            except ValueError as exc:
                raise LibraryError(f"invalid response: {self.response}") from exc
        if not isinstance(self.status, ConstraintStatus):
            try:
                object.__setattr__(self, "status", ConstraintStatus(self.status))
            except ValueError as exc:
                raise LibraryError(f"invalid status: {self.status}") from exc
        object.__setattr__(
            self,
            "source_episode_ids",
            _string_tuple(self.source_episode_ids, "source_episode_ids"),
        )
        if not isinstance(self.metadata, Mapping):
            raise LibraryError("metadata must be a mapping")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SoftConstraint":
        required = {
            "constraint_id",
            "action_types",
            "tactic_type",
            "trigger_pattern",
            "keywords",
            "instruction",
            "response",
            "status",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise LibraryError(f"constraint is missing fields: {', '.join(missing)}")
        return cls(
            constraint_id=payload["constraint_id"],
            action_types=tuple(payload["action_types"]),
            tactic_type=payload["tactic_type"],
            trigger_pattern=payload["trigger_pattern"],
            keywords=tuple(payload["keywords"]),
            instruction=payload["instruction"],
            response=ConstraintResponse(payload["response"]),
            status=ConstraintStatus(payload["status"]),
            source_episode_ids=tuple(payload.get("source_episode_ids", ())),
            metadata=payload.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)

    def approved_copy(self, *, metadata: Mapping[str, Any] | None = None) -> "SoftConstraint":
        merged = dict(self.metadata)
        if metadata:
            merged.update(metadata)
        return replace(self, status=ConstraintStatus.APPROVED, metadata=merged)


@dataclass(frozen=True, slots=True)
class FrozenConstraintLibrary:
    constraints: tuple[SoftConstraint, ...] = ()
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.constraints, tuple):
            raise LibraryError("constraints must be a tuple")
        ids = [item.constraint_id for item in self.constraints]
        duplicates = sorted({item for item in ids if ids.count(item) > 1})
        if duplicates:
            raise LibraryError(f"duplicate constraint IDs: {', '.join(duplicates)}")
        canonical_records = [
            item.to_dict()
            for item in sorted(self.constraints, key=lambda value: value.constraint_id)
        ]
        digest = sha256(canonical_json(canonical_records).encode("utf-8")).hexdigest()
        object.__setattr__(self, "digest", digest)

    @property
    def approved(self) -> tuple[SoftConstraint, ...]:
        return tuple(
            item
            for item in self.constraints
            if item.status is ConstraintStatus.APPROVED
        )

    @classmethod
    def from_iterable(cls, constraints: Iterable[SoftConstraint]) -> "FrozenConstraintLibrary":
        return cls(tuple(constraints))

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "FrozenConstraintLibrary":
        records: list[SoftConstraint] = []
        source = Path(path)
        try:
            with source.open("r", encoding="utf-8") as handle:
                for line_number, raw_line in enumerate(handle, start=1):
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                        if not isinstance(payload, dict):
                            raise LibraryError("record must be an object")
                        records.append(SoftConstraint.from_dict(payload))
                    except (json.JSONDecodeError, LibraryError, KeyError, TypeError, ValueError) as exc:
                        raise LibraryError(f"{source}:{line_number}: {exc}") from exc
        except OSError as exc:
            raise LibraryError(f"could not read library {source}: {exc}") from exc
        return cls.from_iterable(records)

    def to_jsonl(self) -> str:
        ordered = sorted(self.constraints, key=lambda item: item.constraint_id)
        return "".join(
            json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            for item in ordered
        )
