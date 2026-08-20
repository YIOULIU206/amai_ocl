"""Append-only filesystem versions for approved constraint libraries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from .json_utils import jsonable
from .learning import PairedRolloutPromotionPolicy, PromotionPolicy, PromotionResult
from .library import ConstraintStatus, FrozenConstraintLibrary, LibraryError


@dataclass(frozen=True, slots=True)
class LibraryVersion:
    version_id: str
    path: Path
    library: FrozenConstraintLibrary
    manifest: Mapping[str, Any]


class VersionedLibraryStore:
    """Creates immutable child directories; existing versions are never overwritten."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def create_initial(
        self,
        library: FrozenConstraintLibrary | None = None,
        *,
        version_id: str = "L000",
    ) -> LibraryVersion:
        base = library or FrozenConstraintLibrary()
        return self._write_version(
            version_id=version_id,
            library=base,
            manifest={
                "version_id": version_id,
                "parent_version": None,
                "parent_digest": None,
                "library_digest": base.digest,
                "promoted_candidate_id": None,
            },
        )

    def promote(
        self,
        *,
        parent: LibraryVersion,
        result: PromotionResult,
        policy: PromotionPolicy | PairedRolloutPromotionPolicy,
        version_id: str,
    ) -> LibraryVersion:
        if not result.approved or result.constraint.status is not ConstraintStatus.APPROVED:
            raise LibraryError("only approved promotion results create a child version")
        library = FrozenConstraintLibrary(
            parent.library.constraints + (result.constraint,)
        )
        manifest = {
            "version_id": version_id,
            "parent_version": parent.version_id,
            "parent_digest": parent.library.digest,
            "library_digest": library.digest,
            "promoted_candidate_id": result.constraint.constraint_id,
            "validation_report": jsonable(result.report),
            "promotion_policy": jsonable(policy),
        }
        return self._write_version(
            version_id=version_id,
            library=library,
            manifest=manifest,
        )

    def load(self, version_id: str) -> LibraryVersion:
        directory = self.root / version_id
        library = FrozenConstraintLibrary.from_jsonl(directory / "constraints.jsonl")
        try:
            with (directory / "manifest.json").open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except OSError as exc:
            raise LibraryError(f"could not load version manifest: {exc}") from exc
        if manifest.get("library_digest") != library.digest:
            raise LibraryError("version manifest digest does not match library contents")
        return LibraryVersion(version_id, directory, library, manifest)

    def _write_version(
        self,
        *,
        version_id: str,
        library: FrozenConstraintLibrary,
        manifest: Mapping[str, Any],
    ) -> LibraryVersion:
        if not version_id or "/" in version_id or "\\" in version_id or version_id in {".", ".."}:
            raise LibraryError("version_id must be a simple non-empty name")
        self.root.mkdir(parents=True, exist_ok=True)
        directory = self.root / version_id
        try:
            directory.mkdir()
        except FileExistsError as exc:
            raise LibraryError(f"library version already exists: {version_id}") from exc
        completed_manifest = {
            **dict(manifest),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._atomic_write(directory / "constraints.jsonl", library.to_jsonl())
        self._atomic_write(
            directory / "manifest.json",
            json.dumps(completed_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        return LibraryVersion(version_id, directory, library, completed_manifest)

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            handle.write(content)
            temporary = Path(handle.name)
        os.replace(temporary, path)
