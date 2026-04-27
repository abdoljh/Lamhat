# Expose only the high-level pipeline API at package level.
# Individual core classes are importable directly from their modules,
# e.g.:  from phase1.core.chunker import SemanticChunker
# Keeping this minimal prevents import-time crashes when optional
# dependencies (fitz, easyocr, arabic_reshaper …) are not yet installed.

from .pipeline import (  # noqa: F401
    Phase1Config,
    Phase1aResult,
    Phase1aPipeline,
    Phase1bPipeline,
    Phase1Pipeline,
    Phase1Result,
)

__all__ = [
    "Phase1Config",
    "Phase1aResult",
    "Phase1aPipeline",
    "Phase1bPipeline",
    "Phase1Pipeline",
    "Phase1Result",
]
