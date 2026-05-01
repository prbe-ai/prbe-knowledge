"""prbe-knowledge ops scripts (CLI tools, seeders, sync helpers).

This file exists so submodules like `scripts.synth` are unambiguously
importable. Individual scripts (e.g. `scripts.seed_synthetic`) work either
way under Python 3.12's implicit namespace packages, but explicit beats
implicit when nesting subpackages.
"""
