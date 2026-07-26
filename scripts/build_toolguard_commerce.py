#!/usr/bin/env python3
"""Build (or verify) the ToolGuard guards for the ToolGuard-Commerce baseline.

The tool definition in aimai_ocl/baselines/toolguard_commerce/tools.py is the
source of truth: its callables are handed to toolguard.build_toolguards. The
YAML file specs/commerce_tools.yaml is a readable mirror only.

External dependency, pinned:
    IBM/tool_guard @ 20e21db4c275d79f8d7bf33ffb985d0b45f786f5

Typical usage:

    # environment / mirror / plan checks only, no credentials needed
    python scripts/build_toolguard_commerce.py --verify-mirror
    python scripts/build_toolguard_commerce.py --print-plan \
        --out-dir artifacts/toolguard_commerce

    # real guard generation (needs LLM credentials)
    python scripts/build_toolguard_commerce.py \
        --out-dir artifacts/toolguard_commerce \
        --step1-model gpt-4o-2024-08-06 --step1-provider openai

    # preflight of already generated guards
    python scripts/build_toolguard_commerce.py --preflight \
        --out-dir artifacts/toolguard_commerce/step2

This script never invents results: when a required credential, environment
variable, or dependency is missing it stops with a non-zero exit code.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aimai_ocl.baselines.toolguard_commerce import (  # noqa: E402
    COMMERCE_APP_NAME,
    COMMERCE_TOOL_NAME,
    SELLER_POLICY_PATH,
    TOOLGUARD_PINNED_COMMIT,
    TOOLGUARD_PIP_INSTALL_ARG,
    TOOL_SPEC_MIRROR_PATH,
    GuardUnavailableError,
    preflight_report,
    tool_functions,
    tool_spec,
    tool_spec_hash,
)

STEP2_MODEL_ENV = "TOOLGUARD_STEP2_GENAI_MODEL"
STEP2_BACKEND_ENV = "TOOLGUARD_STEP2_GENAI_BACKEND"
BUILD_MANIFEST_FILENAME = "build_manifest.json"


class BuildPreconditionError(RuntimeError):
    """Raised when the build cannot start (missing dependency or credential)."""


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        return "missing"
    return sha256(path.read_bytes()).hexdigest()


def _normalize_text(value: Any) -> Any:
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip()
    if isinstance(value, dict):
        return {key: _normalize_text(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_text(item) for item in value]
    return value


def verify_mirror(spec_path: Path = TOOL_SPEC_MIRROR_PATH) -> tuple[bool, str]:
    """Compare the YAML mirror with the Python tool spec."""
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise BuildPreconditionError(
            "PyYAML is required to verify the spec mirror: pip install pyyaml"
        ) from exc
    if not spec_path.is_file():
        return False, "Spec mirror not found: " + str(spec_path)
    mirrored = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    expected = tool_spec()
    if _normalize_text(mirrored) == _normalize_text(expected):
        return True, "Spec mirror matches tools.py (normalized whitespace)."
    return False, (
        "Spec mirror is out of sync with tools.py.\n--- mirror ---\n"
        + json.dumps(_normalize_text(mirrored), indent=2, sort_keys=True)
        + "\n--- tools.py ---\n"
        + json.dumps(_normalize_text(expected), indent=2, sort_keys=True)
    )


def write_mirror(spec_path: Path = TOOL_SPEC_MIRROR_PATH) -> Path:
    """Regenerate the YAML mirror from tools.py."""
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise BuildPreconditionError(
            "PyYAML is required to write the spec mirror: pip install pyyaml"
        ) from exc
    header = (
        "# Readable mirror of the ToolGuard-Commerce tool specification.\n"
        "#\n"
        "# SOURCE OF TRUTH: aimai_ocl/baselines/toolguard_commerce/tools.py\n"
        "# Regenerate with:\n"
        "#   python scripts/build_toolguard_commerce.py --write-mirror\n\n"
    )
    body = yaml.safe_dump(tool_spec(), sort_keys=False, allow_unicode=True, width=88)
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(header + body, encoding="utf-8")
    return spec_path


def check_build_environment(*, step1_provider: str) -> dict[str, Any]:
    """Verify dependencies and credentials needed for a real guard build."""
    problems: list[str] = []
    details: dict[str, Any] = {
        "toolguard_pinned_commit": TOOLGUARD_PINNED_COMMIT,
        "step1_provider": step1_provider,
    }

    try:
        import toolguard  # noqa: F401
        from toolguard.core import build_toolguards  # noqa: F401

        details["toolguard_importable"] = True
    except Exception as exc:
        details["toolguard_importable"] = False
        problems.append(
            "toolguard is not importable ("
            + repr(exc)
            + "); install the pinned commit with: pip install "
            + repr(TOOLGUARD_PIP_INSTALL_ARG)
        )

    try:
        from toolguard.llm.tg_litellm import LitellmModel  # noqa: F401

        details["litellm_importable"] = True
    except Exception as exc:
        details["litellm_importable"] = False
        problems.append("toolguard.llm.tg_litellm is unavailable: " + repr(exc))

    try:
        import mellea  # noqa: F401

        details["mellea_importable"] = True
    except Exception as exc:
        details["mellea_importable"] = False
        problems.append(
            "mellea is required by the pinned ToolGuard step 2 code generator: "
            + repr(exc)
        )

    step2_model = os.getenv(STEP2_MODEL_ENV)
    details[STEP2_MODEL_ENV] = step2_model
    details[STEP2_BACKEND_ENV] = os.getenv(STEP2_BACKEND_ENV, "openai")
    if not step2_model:
        problems.append(
            STEP2_MODEL_ENV
            + " is not set; the pinned ToolGuard step 2 asserts on it."
        )

    api_key_present = bool(os.getenv("OPENAI_API_KEY"))
    details["OPENAI_API_KEY_present"] = api_key_present
    details["OPENAI_API_BASE"] = os.getenv("OPENAI_API_BASE")
    if not api_key_present:
        problems.append(
            "OPENAI_API_KEY is not set; step 1 and step 2 both need model access."
        )

    details["problems"] = problems
    details["ready"] = not problems
    return details


def build_plan(*, out_dir: Path, policy_path: Path, app_name: str, step1_model: str, step1_provider: str) -> dict[str, Any]:
    """Describe exactly what a build would do (no side effects)."""
    return {
        "toolguard_pinned_commit": TOOLGUARD_PINNED_COMMIT,
        "app_name": app_name,
        "tool_name": COMMERCE_TOOL_NAME,
        "tools_source_of_truth": "aimai_ocl/baselines/toolguard_commerce/tools.py",
        "tool_spec_sha256": tool_spec_hash(),
        "policy_path": str(policy_path),
        "policy_sha256": _sha256_file(policy_path),
        "spec_mirror_path": str(TOOL_SPEC_MIRROR_PATH),
        "spec_mirror_sha256": _sha256_file(TOOL_SPEC_MIRROR_PATH),
        "out_dir": str(out_dir),
        "step1_dir": str(out_dir / "step1"),
        "step2_dir": str(out_dir / "step2"),
        "step1_model": step1_model,
        "step1_provider": step1_provider,
        "step2_backend_env": STEP2_BACKEND_ENV,
        "step2_model_env": STEP2_MODEL_ENV,
    }


def run_build(
    *,
    out_dir: Path,
    policy_path: Path,
    app_name: str,
    step1_model: str,
    step1_provider: str,
    tools2run: list[str] | None,
    short_step1: bool,
) -> dict[str, Any]:
    """Generate the commerce guards with the pinned ToolGuard."""
    env = check_build_environment(step1_provider=step1_provider)
    if not env["ready"]:
        raise BuildPreconditionError(
            "Cannot build ToolGuard guards:\n  - " + "\n  - ".join(env["problems"])
        )
    if not policy_path.is_file():
        raise BuildPreconditionError("Policy file not found: " + str(policy_path))

    from toolguard.core import build_toolguards
    from toolguard.llm.tg_litellm import LitellmModel

    policy_text = policy_path.read_text(encoding="utf-8")
    try:
        import markdown

        policy_payload = markdown.markdown(policy_text)
    except ModuleNotFoundError:
        # The pinned CLI converts markdown to HTML; plain markdown is accepted
        # as-is when the converter is unavailable.
        policy_payload = policy_text

    out_dir.mkdir(parents=True, exist_ok=True)
    llm = LitellmModel(step1_model, step1_provider)
    result = asyncio.run(
        build_toolguards(
            policy_text=policy_payload,
            tools=tool_functions(),
            out_dir=str(out_dir),
            step1_llm=llm,
            app_name=app_name,
            tools2run=tools2run,
            short1=short_step1,
        )
    )
    generated_tools = sorted(getattr(result, "tools", {}) or {})
    if COMMERCE_TOOL_NAME not in generated_tools:
        raise BuildPreconditionError(
            "The build produced no guard for "
            + COMMERCE_TOOL_NAME
            + "; generated: "
            + ", ".join(generated_tools)
        )

    manifest = {
        "toolguard_pinned_commit": TOOLGUARD_PINNED_COMMIT,
        "app_name": app_name,
        "tool_name": COMMERCE_TOOL_NAME,
        "generated_tools": generated_tools,
        "tool_spec_sha256": tool_spec_hash(),
        "policy_path": str(policy_path),
        "policy_sha256": _sha256_file(policy_path),
        "spec_mirror_sha256": _sha256_file(TOOL_SPEC_MIRROR_PATH),
        "step1_model": step1_model,
        "step1_provider": step1_provider,
        "step2_backend": os.getenv(STEP2_BACKEND_ENV, "openai"),
        "step2_model": os.getenv(STEP2_MODEL_ENV),
        "out_dir": str(out_dir),
    }
    manifest_path = out_dir / BUILD_MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    manifest["build_manifest_path"] = str(manifest_path)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="build_toolguard_commerce",
        description="Build or verify ToolGuard-Commerce guards (pinned ToolGuard).",
    )
    parser.add_argument("--out-dir", default="artifacts/toolguard_commerce", help="Output folder for generated artifacts")
    parser.add_argument("--policy", default=str(SELLER_POLICY_PATH), help="Path to the seller policy markdown")
    parser.add_argument("--app-name", default=COMMERCE_APP_NAME, help="ToolGuard app name")
    parser.add_argument("--step1-model", default="gpt-4o-2024-08-06", help="Model used for ToolGuard step 1")
    parser.add_argument("--step1-provider", default="openai", help="litellm provider for step 1, e.g. openai, azure, RITS")
    parser.add_argument("--tools2run", nargs="+", default=None, help="Optional subset of tool names")
    parser.add_argument("--short-step1", action="store_true", help="Run the short version of step 1")
    parser.add_argument("--verify-mirror", action="store_true", help="Check specs/commerce_tools.yaml against tools.py and exit")
    parser.add_argument("--write-mirror", action="store_true", help="Regenerate specs/commerce_tools.yaml from tools.py and exit")
    parser.add_argument("--check-env", action="store_true", help="Report build prerequisites and exit")
    parser.add_argument("--print-plan", action="store_true", help="Print the build plan and exit")
    parser.add_argument("--preflight", action="store_true", help="Preflight already generated guards in --out-dir and exit")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    policy_path = Path(args.policy)

    if args.write_mirror:
        path = write_mirror()
        print("Wrote spec mirror: " + str(path))
        return 0

    if args.verify_mirror:
        ok, message = verify_mirror()
        print(message)
        return 0 if ok else 1

    if args.check_env:
        report = check_build_environment(step1_provider=args.step1_provider)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["ready"] else 1

    if args.print_plan:
        plan = build_plan(
            out_dir=out_dir,
            policy_path=policy_path,
            app_name=args.app_name,
            step1_model=args.step1_model,
            step1_provider=args.step1_provider,
        )
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    if args.preflight:
        try:
            report = preflight_report(out_dir)
        except GuardUnavailableError as exc:
            print("PREFLIGHT FAILED: " + str(exc), file=sys.stderr)
            return 1
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    try:
        manifest = run_build(
            out_dir=out_dir,
            policy_path=policy_path,
            app_name=args.app_name,
            step1_model=args.step1_model,
            step1_provider=args.step1_provider,
            tools2run=args.tools2run,
            short_step1=args.short_step1,
        )
    except BuildPreconditionError as exc:
        print("BUILD STOPPED: " + str(exc), file=sys.stderr)
        return 2
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
