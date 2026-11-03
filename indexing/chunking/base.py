from abc import ABC, abstractmethod
from dataclasses import dataclass, field

@dataclass
class Chunk:
    text: str
    section_label: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)

class ChunkingStrategy(ABC):
    @abstractmethod
    def chunk(self, opinion_text: str, opinion_metadata: dict) -> list[Chunk] :
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...
