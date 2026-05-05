"""Preset loader: resolve a named preset YAML and merge it into a raw profile dict.

Presets supply default values for `archetypes` and `llm` blocks. Profile values
always win over preset values on any key conflict.
"""
from __future__ import annotations

from pathlib import Path

import yaml

_PRESETS_DIR = Path(__file__).parent


class PresetNotFoundError(FileNotFoundError):
    """Raised when the requested preset name does not match any shipped YAML file."""


def apply_preset(profile_raw: dict, preset_name: str | None) -> dict:
    """Load preset YAML from scripts/synth/presets/<preset_name>.yaml.

    Merge the preset values into profile_raw (preset is the BASE, profile
    values OVERRIDE preset values). Specifically:
    - Preset's `archetypes` block is merged into profile_raw['archetypes']
      (per-archetype dict merge; profile wins on conflict).
    - Preset's `llm` block is merged into profile_raw.get('llm', {});
      profile wins on conflict.
    - Other top-level keys are NOT merged (presets only set archetypes + llm).

    Returns the merged dict. profile_raw is not mutated.
    Raises PresetNotFoundError if the preset name does not match a shipped file.
    Returns profile_raw unchanged if preset_name is None or empty string.
    Empty/falsy `archetypes` or `llm` blocks in the preset are skipped (not
    merged as empty dicts).
    """
    if not preset_name:
        return profile_raw

    preset_file = _PRESETS_DIR / f"{preset_name}.yaml"
    if not preset_file.exists():
        raise PresetNotFoundError(
            f"Preset {preset_name!r} not found. "
            f"Expected file: {preset_file}. "
            f"Available presets: {[p.stem for p in _PRESETS_DIR.glob('*.yaml')]}"
        )

    with preset_file.open() as f:
        preset = yaml.safe_load(f) or {}

    merged = dict(profile_raw)

    preset_archetypes = preset.get("archetypes", {})
    if preset_archetypes:
        merged_archetypes = dict(preset_archetypes)
        for arch_name, arch_cfg in profile_raw.get("archetypes", {}).items():
            if arch_name in merged_archetypes:
                merged_archetypes[arch_name] = {**merged_archetypes[arch_name], **arch_cfg}
            else:
                merged_archetypes[arch_name] = arch_cfg
        merged["archetypes"] = merged_archetypes

    preset_llm = preset.get("llm", {})
    if preset_llm:
        merged_llm = dict(preset_llm)
        merged_llm.update(profile_raw.get("llm", {}))
        merged["llm"] = merged_llm

    return merged
