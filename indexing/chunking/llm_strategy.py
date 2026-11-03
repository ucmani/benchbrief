import json
from .base import Chunk, ChunkingStrategy

SECTION_LABELS = ["facts", "legal_standard", "analysis", "holding", "other"]

CLASSIFICATION_PROMPT_TEMPLATE = """You are analyzing a legal opinion. Split the following opinion text into sections, where each section is labeled with exactly one of: {labels}.

Return ONLY valid JSON: a list of objects, each with "label" and "text" keys, in the order they appear in the document. Do not include any commentary outside the JSON.

Opinion text:
{opinion_text}
"""

class LLMStrategy(ChunkingStrategy):
    def __init__(self, llm_call_fn):
        self.llm_call_fn = llm_call_fn

    @property
    def name(self) -> str:
        return "llm"

    def chunk(self, opinion_text: str, opinion_metadata: dict) -> list[Chunk]:
        prompt = CLASSIFICATION_PROMPT_TEMPLATE.format(
            labels=", ".join(SECTION_LABELS),
            opinion_text=opinion_text,
        )

        response_text = self.llm_call_fn(prompt)

        try:
            sections = json.loads(response_text)

        except json.JSONDecodeError:
            return [
                Chunk(
                    text=opinion_text,
                    section_label="other",
                    chunk_index=0,
                    metadata=opinion_metadata
                )
            ]

        chunks = []

        for i, section in enumerate(sections):
            label = section.get("label","other")
            if label not in SECTION_LABELS:
                label = "other"

            chunks.append(
                Chunk(
                    text=section.get("text",""),
                    section_label=label,
                    chunk_index=i,
                    metadata=opinion_metadata
                )
            )

        return chunks