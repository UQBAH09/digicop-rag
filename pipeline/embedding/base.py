from abc import ABC, abstractmethod

from shared.models import Chunk

class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, chunks: list[Chunk]) -> list[list[float]]:
        """
        Embed a list of chunks and return one vector per chunk.

        The returned list is guaranteed to be the same length as the input
        and in the same order — vectors[i] corresponds to chunks[i].
        The indexer relies on this ordering when it zips chunks with vectors.
        """
        ...
    
    @property
    @abstractmethod
    def dim(self) -> int: 
        """
        The vector dimension this embedder produces.
        Used by the indexer to size the Qdrant collection correctly.
        """
        ...

    @property
    @abstractmethod
    def model_id(self) -> str:
        """
        A string identifying the model and version, e.g. 'bge-m3'.
        Stored on the Qdrant collection so a mismatch fails loudly
        if someone runs the indexer against a collection built with
        a different model.
        """
        ...