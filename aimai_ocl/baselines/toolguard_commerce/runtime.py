"""Load and invoke pinned ToolGuard generated guards, failing closed.

External dependency, pinned:
    IBM/tool_guard @ 20e21db4c275d79f8d7bf33ffb985d0b45f786f5

Two hard rules are implemented here:

1. Guards are never optional at run time. If the generated guard directory is
   missing, unreadable, does not contain a guard for the commerce tool, or the
   pinned toolguard package cannot be imported, loading raises
   GuardUnavailableError and the caller must stop. Nothing is silently
   allowed through.
2. Guards may not perform auxiliary API lookups. NoLookupInvoker answers every
   invoke() with UnsupportedGuardLookupError instead of fabricating data or
   returning None, so a guard that needs extra tool calls fails closed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable

from aimai_ocl.baselines.toolguard_commerce.tools import COMMERCE_TOOL_NAME

TOOLGUARD_REPO_URL = "https://github.com/IBM/tool_guard"
TOOLGUARD_PINNED_COMMIT = "20e21db4c275d79f8d7bf33ffb985d0b45f786f5"
TOOLGUARD_PIP_SPEC = "toolguard @ git+" + TOOLGUARD_REPO_URL + "@" + TOOLGUARD_PINNED_COMMIT
TOOLGUARD_PIP_INSTALL_ARG = "git+" + TOOLGUARD_REPO_URL + "@" + TOOLGUARD_PINNED_COMMIT

GUARD_RESULT_FILENAME = "result.json"
_POLICY_VIOLATION_CLASS_NAME = "PolicyViolationException"
_HASHED_SUFFIXES = (".py", ".json", ".md", ".yaml", ".yml")

BUILD_HINT = (
    "Generate the guards first: python scripts/build_toolguard_commerce.py "
    "--out-dir artifacts/toolguard_commerce"
)


class ToolGuardCommerceError(RuntimeError):
    """Base error for the ToolGuard-Commerce baseline."""


class GuardUnavailableError(ToolGuardCommerceError):
    """Raised when generated guards are missing or cannot be loaded."""


class GuardExecutionError(ToolGuardCommerceError):
    """Raised when a guard fails for a reason other than a policy violation."""


class UnsupportedGuardLookupError(ToolGuardCommerceError):
    """Raised when a guard tries to perform an auxiliary API lookup.

    The ToolGuard-Commerce baseline exposes no auxiliary API to guards. Any
    invoke() attempt is an explicit, loud failure: returning fake data or None
    would silently weaken the guard.
    """

    def __init__(self, toolname: str, arguments: Any = None) -> None:
        self.toolname = toolname
        self.arguments = arguments
        super().__init__(
            "Guard requested an auxiliary tool lookup for "
            + repr(toolname)
            + ", but the ToolGuard-Commerce baseline provides no lookup API. "
            "The evaluation fails closed instead of returning synthetic data. "
            "Regenerate guards that decide from their arguments only."
        )


class NoLookupInvoker:
    """Tool invoker that refuses every auxiliary lookup."""

    def invoke(self, toolname: str, arguments: Dict[str, Any] | None = None, model: Any = None) -> Any:
        """Always raise UnsupportedGuardLookupError."""
        raise UnsupportedGuardLookupError(toolname=toolname, arguments=arguments)


def _register_invoker_base() -> bool:
    """Register NoLookupInvoker as a virtual subclass of IToolInvoker."""
    try:
        from toolguard.runtime import IToolInvoker
    except Exception:  # pragma: no cover - toolguard is an optional extra
        return False
    IToolInvoker.register(NoLookupInvoker)
    return True


NO_LOOKUP_INVOKER_REGISTERED = _register_invoker_base()


def is_policy_violation(exc: BaseException) -> bool:
    """Return True when exc is a ToolGuard PolicyViolationException.

    Generated guards import PolicyViolationException from the *copied* runtime
    package inside the generated guard directory, so the class object differs
    from toolguard.data_types.PolicyViolationException. Detection is therefore
    by class name across the exception MRO.
    """
    for klass in type(exc).__mro__:
        if klass.__name__ == _POLICY_VIOLATION_CLASS_NAME:
            return True
    return False


def policy_violation_message(exc: BaseException) -> str:
    """Extract the raw violation message from a guard exception."""
    message = getattr(exc, "message", None)
    if isinstance(message, str) and message.strip():
        return message.strip()
    text = str(exc).strip()
    return text or "Policy violation reported without a message."


def toolguard_import_status() -> dict[str, Any]:
    """Report whether the pinned toolguard package is importable."""
    status: dict[str, Any] = {
        "pinned_commit": TOOLGUARD_PINNED_COMMIT,
        "repo_url": TOOLGUARD_REPO_URL,
        "pip_spec": TOOLGUARD_PIP_SPEC,
        "importable": False,
        "module_path": None,
        "distribution_version": None,
        "import_error": None,
    }
    try:
        import toolguard
        import toolguard.runtime as toolguard_runtime
    except Exception as exc:
        status["import_error"] = repr(exc)
        return status
    status["importable"] = True
    status["module_path"] = getattr(toolguard, "__file__", None)
    status["runtime_module_path"] = getattr(toolguard_runtime, "__file__", None)
    try:
        from importlib.metadata import version

        status["distribution_version"] = version("toolguard")
    except Exception:  # pragma: no cover - metadata may be absent
        status["distribution_version"] = None
    return status


def guard_directory_hash(directory: str | Path) -> str:
    """Hash the generated guard directory deterministically."""
    root = Path(directory)
    if not root.is_dir():
        raise GuardUnavailableError(
            "Generated guard directory not found: " + str(root) + ". " + BUILD_HINT
        )
    digest = sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.suffix.lower() not in _HASHED_SUFFIXES:
            continue
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\x00")
        digest.update(path.read_bytes())
        digest.update(b"\x00")
    return digest.hexdigest()


def read_guard_result(
    directory: str | Path,
    *,
    filename: str = GUARD_RESULT_FILENAME,
) -> dict[str, Any]:
    """Read and validate the generated ToolGuard result payload."""
    root = Path(directory)
    if not root.is_dir():
        raise GuardUnavailableError(
            "Generated guard directory not found: " + str(root) + ". " + BUILD_HINT
        )
    result_path = root / filename
    if not result_path.is_file():
        raise GuardUnavailableError(
            "Generated guard result file not found: " + str(result_path) + ". " + BUILD_HINT
        )
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise GuardUnavailableError(
            "Generated guard result file is unreadable: " + str(result_path)
        ) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("tools"), dict):
        raise GuardUnavailableError(
            "Generated guard result file has no tools mapping: " + str(result_path)
        )
    return payload


def _guard_file_for_tool(payload: dict[str, Any], tool_name: str) -> str:
    tools = payload.get("tools") or {}
    entry = tools.get(tool_name)
    if not isinstance(entry, dict):
        raise GuardUnavailableError(
            "No generated guard for tool " + repr(tool_name) + "; generated tools: "
            + ", ".join(sorted(str(key) for key in tools))
            + ". " + BUILD_HINT
        )
    guard_file = (entry.get("guard_file") or {}).get("file_name")
    if not guard_file:
        raise GuardUnavailableError(
            "Generated guard entry for " + repr(tool_name) + " has no guard file name."
        )
    return str(guard_file)


def load_commerce_guards(
    directory: str | Path,
    *,
    filename: str = GUARD_RESULT_FILENAME,
    tool_name: str = COMMERCE_TOOL_NAME,
) -> Any:
    """Load the generated ToolGuard runtime for the commerce tool.

    Output:
        A toolguard ToolguardRuntime exposing
        check_toolcall(tool_name, args, delegate).

    Raises:
        GuardUnavailableError: If anything is missing. The baseline stops
            instead of running unguarded.
    """
    root = Path(directory)
    payload = read_guard_result(root, filename=filename)
    guard_file = _guard_file_for_tool(payload, tool_name)
    if not (root / guard_file).is_file():
        raise GuardUnavailableError(
            "Generated guard module is missing on disk: " + str(root / guard_file) + ". " + BUILD_HINT
        )
    status = toolguard_import_status()
    if not status["importable"]:
        raise GuardUnavailableError(
            "The pinned toolguard package is not importable ("
            + str(status["import_error"])
            + "). Install it with: pip install "
            + repr(TOOLGUARD_PIP_INSTALL_ARG)
        )
    from toolguard.runtime import load_toolguards

    try:
        return load_toolguards(str(root), filename)
    except Exception as exc:
        raise GuardUnavailableError(
            "Failed to load generated guards from " + str(root) + ": " + repr(exc)
        ) from exc


@dataclass
class PreflightReport:
    """Result of the generated-guard preflight check."""

    guard_dir: str
    tool_name: str
    guard_file: str
    guard_dir_hash: str
    toolguard: dict[str, Any] = field(default_factory=dict)
    no_lookup_invoker_fails_closed: bool = False
    guard_loaded: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "guard_dir": self.guard_dir,
            "tool_name": self.tool_name,
            "guard_file": self.guard_file,
            "guard_dir_hash": self.guard_dir_hash,
            "toolguard": dict(self.toolguard),
            "no_lookup_invoker_fails_closed": self.no_lookup_invoker_fails_closed,
            "guard_loaded": self.guard_loaded,
        }


def _selfcheck_no_lookup_invoker() -> bool:
    """Verify that NoLookupInvoker raises instead of returning data."""
    try:
        NoLookupInvoker().invoke("any_tool", {"probe": True}, None)
    except UnsupportedGuardLookupError:
        return True
    return False


def preflight_report(
    directory: str | Path,
    *,
    filename: str = GUARD_RESULT_FILENAME,
    tool_name: str = COMMERCE_TOOL_NAME,
    load_runtime: bool = True,
) -> dict[str, Any]:
    """Run the generated-guard preflight, raising on any missing piece."""
    root = Path(directory)
    payload = read_guard_result(root, filename=filename)
    guard_file = _guard_file_for_tool(payload, tool_name)
    if not (root / guard_file).is_file():
        raise GuardUnavailableError(
            "Generated guard module is missing on disk: " + str(root / guard_file)
        )
    if not _selfcheck_no_lookup_invoker():  # pragma: no cover - defensive
        raise GuardUnavailableError(
            "NoLookupInvoker did not fail closed; refusing to run the baseline."
        )
    report = PreflightReport(
        guard_dir=str(root),
        tool_name=tool_name,
        guard_file=guard_file,
        guard_dir_hash=guard_directory_hash(root),
        toolguard=toolguard_import_status(),
        no_lookup_invoker_fails_closed=True,
    )
    if load_runtime:
        load_commerce_guards(root, filename=filename, tool_name=tool_name)
        report.guard_loaded = True
    return report.to_dict()


def hash_files(paths: Iterable[str | Path]) -> dict[str, str]:
    """Return sha256 digests for the given files (missing files -> None)."""
    digests: dict[str, str] = {}
    for raw in paths:
        path = Path(raw)
        if path.is_file():
            digests[str(path)] = sha256(path.read_bytes()).hexdigest()
        else:
            digests[str(path)] = "missing"
    return digests
