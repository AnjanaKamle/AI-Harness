"""Scripted mode: the deterministic scripted client (``AI_PROVIDER=scripted`` or ``demo``).

This is not a language model. It is selected only explicitly, and a failing real provider
never falls back to it.
"""

from __future__ import annotations

import os
from typing import Any

from harness.llm.client import LLMClient


def create(settings: Any) -> LLMClient:
    from harness.demo import DEMO_STEP_ENV, DemoScriptedClient

    try:
        step = float(os.environ.get(DEMO_STEP_ENV, "0.4"))
    except ValueError:
        step = 0.4
    return DemoScriptedClient(max(step, 0.0))
