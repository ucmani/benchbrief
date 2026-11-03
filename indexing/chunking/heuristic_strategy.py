import re

from .base import Chunk, ChunkingStrategy

SECTION_PATTERNS: dict[str, re.Pattern] = {
    "facts": re.compile(
        r"^\s*(?:[IVX]+\.|\w\.)?\s*(FACTUAL\s*(?:&|AND)?\s*(?:PROCEDURAL)?\s*BACKGROUND|FACTS|BACKGROUND)\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    "legal_standard": re.compile(
        r"^\s*(?:[IVX]+\.|\w\.)?\s*(LEGAL STANDARD|STANDARD OF REVIEW)\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    "analysis": re.compile(
        r"^\s*(?:[IVX]+\.|\w\.)?\s*(DISCUSSION|ANALYSIS)\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    "holding": re.compile(
        r"^\s*(?:[IVX]+\.|\w\.)?\s*(CONCLUSION|HOLDING|ORDER)\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
}

class HeuristicStrategy(ChunkingStrategy):
    @property
    def name(self) -> str:
        return "heuristic"

    def chunk(self, opinion_text: str, opinion_metadata: dict) -> list[Chunk]:
        matches = []
        for label, pattern in SECTION_PATTERNS.items():
            for m in pattern.finditer(opinion_text):
                matches.append((m.start(), label))

        matches.sort(key=lambda x: x[0])

        chunks = []
        for i, (start_pos, label) in enumerate(matches):
            end_pos = matches[i+1][0] if i < len(matches)-1 else len(opinion_text)
            section_text = opinion_text[start_pos:end_pos].strip()

            if section_text:
                chunks.append(
                    Chunk(
                        text=section_text,
                        section_label=label,
                        chunk_index=i,
                        metadata=opinion_metadata,
                    )
                )

        return chunks