import hashlib
import random

from pipeline.embedding.base import BaseEmbedder
from shared.models import Chunk

class FakeEmbedder(BaseEmbedder):
    def embed(self, chunks: list[Chunk]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for chunk in chunks:
            seed = int(hashlib.sha256(chunk.content.encode('utf-8')).hexdigest(), 16)
            embeddings.append([random.Random(seed).uniform(-1,1) for _ in range(self.dim)])

        return embeddings
            

    @property
    def dim(self) -> int:
        return 1024

    @property
    def model_id(self) -> str:
        return "fake-test"