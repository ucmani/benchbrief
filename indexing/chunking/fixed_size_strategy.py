from .base import Chunk, ChunkingStrategy

DEFAULT_CHUNK_SIZE_WORDS = 400
DEFAULT_OVERLAP_WORDS = 50

class FixedSizeStrategy(ChunkingStrategy):
    def __init__(self, chunk_size_words:int=DEFAULT_CHUNK_SIZE_WORDS, overlap_words: int=DEFAULT_OVERLAP_WORDS):
        self.chunk_size_words = chunk_size_words
        self.overlap_words = overlap_words

    @property
    def name(self) -> str:
        return "fixed_size"

    def chunk(self, opinion_text: str, opinion_metadata: dict) -> list[Chunk]:
        words = opinion_text.split()
        chunks = []
        step = self.chunk_size_words - self.overlap_words
        chunk_index = 0
        i = 0

        while i < len(words):
            window = words[i:i+self.chunk_size_words]
            text = " ".join(window)

            chunks.append(
                Chunk(
                    text=text,
                    section_label="unknown",
                    chunk_index=chunk_index,
                    metadata=opinion_metadata,
                )
            )

            chunk_index += 1
            i += step
        return chunks