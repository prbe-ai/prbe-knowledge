"""Re-export shim: SynthDoc and Source at a stable top-level path.

Downstream code (eval artifact writers, tests) can import:
    from scripts.synth.synth_doc import SynthDoc, Source
"""

from __future__ import annotations

from scripts.synth.archetypes.base import Source
from scripts.synth.output.base import SynthDoc

__all__ = ["Source", "SynthDoc"]
