"""
Concrete embedder implementation using the BAAI/bge-m3 model.

bge-m3 is a multilingual model supporting 100+ languages including Urdu,
which is the primary reason it was chosen over alternatives. It runs locally
with no API cost.
"""

from shared.models import Chunk
from pipeline.embedding.base import BaseEmbedder
from shared.config import embedding_settings


class BgeM3Embedder(BaseEmbedder):

    def __init__(self, batch_size: int = embedding_settings.batch_size):
        # Heavy import lives here, not at module top. FlagEmbedding pulls in
        # PyTorch which takes several seconds to import. Keeping it lazy means
        # tests that use FakeEmbedder don't pay that cost.
        from FlagEmbedding import BGEM3FlagModel

        self._model = BGEM3FlagModel(embedding_settings.model_name, use_fp16=embedding_settings.use_fp16)
        self._batch_size = batch_size

    @property
    def dim(self) -> int:
        return 1024

    @property
    def model_id(self) -> str:
        return "bge-m3"

    def embed(self, chunks: list[Chunk]) -> list[list[float]]:
        content = [chunk.content for chunk in chunks]

        # bge-m3 does NOT require separate query vs document prompts, unlike
        # some E5 variants. The same encode() call is used for both chunks at
        # ingestion time and queries at retrieval time.
        result = self._model.encode(content, batch_size=self._batch_size)

        # encode() returns dense, sparse (lexical_weights), and multi-vector
        # (colbert_vecs) outputs simultaneously. We take only dense_vecs for v1.
        # Sparse and multi-vector retrieval are v2 work.
        return result["dense_vecs"].tolist()