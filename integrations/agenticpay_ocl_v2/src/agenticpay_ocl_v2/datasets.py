"""Stable profile IDs and checked split-manifest validation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence


class DatasetError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Profile:
    profile_id: str
    persona_type: str
    name: str
    description: str
    metadata: Mapping[str, Any]


def load_profiles(path: str | Path) -> tuple[Profile, ...]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetError(f"could not load profiles: {exc}") from exc
    if not isinstance(payload, list):
        raise DatasetError("profile dataset must be a JSON list")
    counters: dict[str, int] = {}
    profiles: list[Profile] = []
    ids: set[str] = set()
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise DatasetError(f"profile {index} must be an object")
        persona = str(item.get("persona_type", "")).strip()
        name = str(item.get("name", "")).strip()
        description = str(item.get("description", "")).strip()
        if not persona or not name or not description:
            raise DatasetError(f"profile {index} is missing persona_type, name, or description")
        counters[persona] = counters.get(persona, 0) + 1
        profile_id = str(
            item.get("profile_id") or f"{persona}_{counters[persona]:03d}"
        )
        if profile_id in ids:
            raise DatasetError(f"duplicate profile ID: {profile_id}")
        ids.add(profile_id)
        profiles.append(
            Profile(
                profile_id=profile_id,
                persona_type=persona,
                name=name,
                description=description,
                metadata=MappingProxyType(
                    {
                        key: value
                        for key, value in item.items()
                        if key not in {"profile_id", "persona_type", "name", "description"}
                    }
                ),
            )
        )
    return tuple(profiles)


@dataclass(frozen=True, slots=True)
class SplitManifest:
    split_id: str
    derivation: tuple[str, ...]
    validation: tuple[str, ...]
    evaluation: tuple[str, ...]
    benign: tuple[str, ...] = ()
    controls: tuple[str, ...] = ()

    @classmethod
    def from_json(cls, path: str | Path) -> "SplitManifest":
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DatasetError(f"could not load split manifest: {exc}") from exc
        try:
            return cls(
                split_id=str(payload["split_id"]),
                derivation=tuple(payload["derivation"]),
                validation=tuple(payload["validation"]),
                evaluation=tuple(payload["evaluation"]),
                benign=tuple(payload.get("benign", ())),
                controls=tuple(payload.get("controls", ())),
            )
        except (KeyError, TypeError) as exc:
            raise DatasetError(f"invalid split manifest: {exc}") from exc

    def validate(self, profiles: Sequence[Profile]) -> None:
        known = {profile.profile_id for profile in profiles}
        named = {
            "derivation": self.derivation,
            "validation": self.validation,
            "evaluation": self.evaluation,
            "benign": self.benign,
            "controls": self.controls,
        }
        seen: dict[str, str] = {}
        for split, ids in named.items():
            for profile_id in ids:
                if profile_id not in known:
                    raise DatasetError(f"unknown profile ID in {split}: {profile_id}")
                if profile_id in seen:
                    raise DatasetError(
                        f"profile {profile_id} appears in both {seen[profile_id]} and {split}"
                    )
                seen[profile_id] = split

    def split_for(self, profile_id: str) -> str | None:
        for name in ("derivation", "validation", "evaluation", "benign", "controls"):
            if profile_id in getattr(self, name):
                return name
        return None
