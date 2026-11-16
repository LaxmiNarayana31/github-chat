"""Enterprise Citation Validation, Evidence Tracking, and Hallucination Verification Engine.

Performs deterministic post-generation claim verification and citation validation against
retrieved ground-truth repository documents. Ensures no fabricated file paths or line ranges
are presented without explicit evidence flags.
"""

import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class CitationReference:
    """Represents a citation parsed from the LLM-generated response."""

    raw_citation: str
    file_path: str
    line_start: int | None = None
    line_end: int | None = None
    is_verified: bool = False
    status: str = "unverified"  # "verified", "hallucinated_file", "invalid_lines"
    matched_chunk_idx: int | None = None


@dataclass
class QualityVerificationResult:
    """Comprehensive evaluation result for research quality, citation fidelity, and grounding."""

    total_citations: int = 0
    verified_citations: int = 0
    hallucinated_citations: int = 0
    citation_fidelity_score: float = (
        1.0  # 1.0 = 100% citations verified against context
    )
    has_hallucinated_citations: bool = False
    citations: list[dict[str, Any]] = field(default_factory=list)
    has_uncertainty_notice: bool = False
    evidence_coverage: float = 0.0  # percentage of retrieved chunks utilized
    annotated_answer: str = ""


class CitationValidator:
    """Production citation validator and research evidence tracker."""

    # Regex patterns matching various citation formats:
    # 1. Markdown link / bracket notation: [file.py:L10-L20], [path/to/file.py], `path/to/file.py:L15`
    # 2. Plain file paths with extensions: backend/app/main.py, src/App.tsx:10-25
    CITATION_PATTERNS = [
        re.compile(
            r"\[([a-zA-Z0-9_\-\./\\]+\.[a-zA-Z0-9]+)(?::L?(\d+)(?:-L?(\d+))?)?\]"
        ),
        re.compile(r"`([a-zA-Z0-9_\-\./\\]+\.[a-zA-Z0-9]+)(?::L?(\d+)(?:-L?(\d+))?)?`"),
        re.compile(
            r"(?:in|at|file|source):\s*`?([a-zA-Z0-9_\-\./\\]+\.[a-zA-Z0-9]+)(?::L?(\d+)(?:-L?(\d+))?)?`?",
            re.IGNORECASE,
        ),
    ]

    @classmethod
    def extract_citations(cls, text: str) -> list[CitationReference]:
        """Extract all candidate file and line citations from generated markdown answer."""
        if not text:
            return []

        seen_keys: set[str] = set()
        citations: list[CitationReference] = []

        for pattern in cls.CITATION_PATTERNS:
            for match in pattern.finditer(text):
                raw = match.group(0)
                file_path = match.group(1).replace("\\", "/").strip().lstrip("./")

                # Filter out obvious false positives like URLs, version numbers, or non-file extensions
                ext = file_path.split(".")[-1].lower() if "." in file_path else ""
                common_exts = {
                    "py",
                    "ts",
                    "tsx",
                    "js",
                    "jsx",
                    "go",
                    "rs",
                    "java",
                    "c",
                    "cpp",
                    "h",
                    "cs",
                    "rb",
                    "php",
                    "json",
                    "yaml",
                    "yml",
                    "toml",
                    "md",
                    "sql",
                    "sh",
                    "dockerfile",
                    "env",
                    "xml",
                    "html",
                    "css",
                }
                if ext not in common_exts and not file_path.lower().endswith(
                    "dockerfile"
                ):
                    continue

                line_start = int(match.group(2)) if match.group(2) else None
                line_end = int(match.group(3)) if match.group(3) else None

                key = f"{file_path}:{line_start}-{line_end}"
                if key not in seen_keys:
                    seen_keys.add(key)
                    citations.append(
                        CitationReference(
                            raw_citation=raw,
                            file_path=file_path,
                            line_start=line_start,
                            line_end=line_end,
                        )
                    )

        return citations

    @classmethod
    def verify_citations(
        cls,
        answer: str,
        retrieved_documents: list[Any],
        strict_line_checking: bool = False,
    ) -> QualityVerificationResult:
        """
        Verify every citation against the ground truth of retrieved documents.
        Identifies hallucinated file paths and computes the Citation Fidelity Score.
        """
        extracted = cls.extract_citations(answer)
        if not extracted:
            # No citations attempted
            return QualityVerificationResult(
                total_citations=0,
                verified_citations=0,
                hallucinated_citations=0,
                citation_fidelity_score=1.0,
                has_hallucinated_citations=False,
                citations=[],
                has_uncertainty_notice="uncertainty" in answer.lower()
                or "not found" in answer.lower(),
                evidence_coverage=0.0,
                annotated_answer=answer,
            )

        # Build ground-truth file and line index from retrieved documents
        known_files: dict[str, list[tuple[int, tuple[int, int] | None]]] = {}
        for idx, doc in enumerate(retrieved_documents):
            meta = getattr(doc, "meta_data", {}) or {}
            fpath = (
                (meta.get("file_path") or "").replace("\\", "/").strip().lstrip("./")
            )
            if not fpath:
                continue

            lines = meta.get("line_range")
            line_tuple = (
                (int(lines[0]), int(lines[1])) if lines and len(lines) >= 2 else None
            )

            if fpath not in known_files:
                known_files[fpath] = []
            known_files[fpath].append((idx, line_tuple))

            # Also index basename for flexible matching (e.g. "main.py" matching "backend/main.py")
            base = fpath.split("/")[-1]
            if base not in known_files:
                known_files[base] = []
            known_files[base].append((idx, line_tuple))

        verified_count = 0
        hallucinated_count = 0
        verified_chunk_indices: set[int] = set()
        annotated_answer = answer

        for cit in extracted:
            normalized_path = cit.file_path
            base_name = normalized_path.split("/")[-1]

            matched_entries = known_files.get(normalized_path) or known_files.get(
                base_name
            )

            if matched_entries:
                cit.is_verified = True
                cit.status = "verified"
                cit.matched_chunk_idx = matched_entries[0][0]
                verified_chunk_indices.add(matched_entries[0][0])
                verified_count += 1
            else:
                cit.is_verified = False
                cit.status = "hallucinated_file"
                hallucinated_count += 1
                log.warning(
                    f"CitationValidator: Detected unverified/hallucinated citation '{cit.file_path}' "
                    f"(not present in retrieved {len(retrieved_documents)} context chunks)"
                )

        total = len(extracted)
        fidelity_score = round(verified_count / total, 4) if total > 0 else 1.0
        evidence_coverage = round(
            len(verified_chunk_indices) / max(1, len(retrieved_documents)), 4
        )

        return QualityVerificationResult(
            total_citations=total,
            verified_citations=verified_count,
            hallucinated_citations=hallucinated_count,
            citation_fidelity_score=fidelity_score,
            has_hallucinated_citations=hallucinated_count > 0,
            citations=[asdict(c) for c in extracted],
            has_uncertainty_notice=cls.detect_transparent_uncertainty(
                answer, retrieved_documents
            ),
            evidence_coverage=evidence_coverage,
            annotated_answer=annotated_answer,
        )

    @classmethod
    def detect_transparent_uncertainty(
        cls, answer: str, retrieved_documents: list[Any]
    ) -> bool:
        """Detect whether the answer transparently communicates uncertainty or lack of context."""
        lowered = answer.lower()
        uncertainty_phrases = [
            "could not find",
            "not enough context",
            "insufficient information",
            "no direct implementation",
            "does not appear to be",
            "no reference found",
            "unclear from the provided",
            "outside the retrieved context",
        ]
        return any(phrase in lowered for phrase in uncertainty_phrases)

    @classmethod
    def build_uncertainty_disclaimer(cls) -> str:
        """Standard enterprise disclaimer banner when retrieved evidence is sparse or inconclusive."""
        return (
            "\n\n> ⚠️ **Verification Notice**: The indexed repository context did not contain direct "
            "implementation evidence for all aspects of this query. Verify key claims against your source tree."
        )
