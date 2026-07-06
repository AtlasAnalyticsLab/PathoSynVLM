"""PathoSynVLM package."""

__all__ = ["PathoSynVLM", "VisionAligner", "load_aligner_from_checkpoint"]


def __getattr__(name: str):
    if name in __all__:
        from .model import PathoSynVLM, VisionAligner, load_aligner_from_checkpoint

        exports = {
            "PathoSynVLM": PathoSynVLM,
            "VisionAligner": VisionAligner,
            "load_aligner_from_checkpoint": load_aligner_from_checkpoint,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
