"""External baselines evaluated against the OCL pipeline.

Sub-packages under aimai_ocl.baselines implement *external* baselines. They
are deliberately isolated from the OCL control layer: nothing in this package
may import aimai_ocl.control, so an external baseline can never inherit OCL
multi-level decisions, deterministic repair, or escalation.
"""

from __future__ import annotations

__all__ = ["toolguard_commerce"]
