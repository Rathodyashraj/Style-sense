"""
src/fusion/__init__.py
───────────────────────
Scoring fusion sub-package (Module 4).

Public exports
--------------
PairwiseScorer    — constructs pairwise feature vectors from explicit + latent.
ColorRuleScorer   — deterministic colour harmony rule engine (complementary,
                    analogous, triadic, split-complementary, neutral).
HarmonyScorer     — top-level pipeline orchestrator; returns HarmonyResult.
HarmonyResult     — dataclass bundling all outputs of a scoring call.
"""

from src.fusion.pairwise_scorer   import PairwiseScorer
from src.fusion.color_rule_scorer import ColorRuleScorer
from src.fusion.harmony_scorer    import HarmonyScorer, HarmonyResult

__all__ = [
    "PairwiseScorer",
    "ColorRuleScorer",
    "HarmonyScorer",
    "HarmonyResult",
]
