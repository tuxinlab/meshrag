"""同一 file_path 多条 doc_status 记录时的权威记录选择测试。

优先级：processed > processing/pending/preprocessed > failed；
同级取 updated_at 最新。此逻辑支撑 Java 中台“删除后自愈”与重新解析判断。
"""

from uuid import uuid4

import numpy as np
import pytest

from lightrag.base import DocStatus, select_authoritative_doc
from lightrag.utils import EmbeddingFunc, Tokenizer

pytestmark = pytest.mark.offline


# ──────────────────────────────────────────────
# 纯函数：权威记录选择
# ──────────────────────────────────────────────


class TestSelectAuthoritativeDoc:
    """验证权威度排序规则。"""

    def test_processed_beats_newer_in_progress_and_failed(self):
        """processed 优先于时间更新的进行中/失败记录。"""
        docs = [
            {
                "id": "failed-newest",
                "status": "failed",
                "updated_at": "2026-01-04T00:00:00+00:00",
            },
            {
                "id": "pending-newer",
                "status": "pending",
                "updated_at": "2026-01-03T00:00:00+00:00",
            },
            {
                "id": "processed-oldest",
                "status": "processed",
                "updated_at": "2026-01-01T00:00:00+00:00",
            },
        ]
        assert select_authoritative_doc(docs)["id"] == "processed-oldest"

    def test_in_progress_beats_failed(self):
        """进行中记录优先于更新的失败记录。"""
        docs = [
            {
                "id": "failed-newest",
                "status": "failed",
                "updated_at": "2026-01-04T00:00:00+00:00",
            },
            {
                "id": "processing-older",
                "status": "processing",
                "updated_at": "2026-01-01T00:00:00+00:00",
            },
        ]
        assert select_authoritative_doc(docs)["id"] == "processing-older"

    @pytest.mark.parametrize("status", ["pending", "processing", "preprocessed"])
    def test_same_tier_takes_latest_updated_at(self, status):
        """同级记录取 updated_at 最新的一条。"""
        docs = [
            {
                "id": "older",
                "status": status,
                "updated_at": "2026-01-01T00:00:00+00:00",
            },
            {
                "id": "newer",
                "status": status,
                "updated_at": "2026-01-02T00:00:00+00:00",
            },
        ]
        assert select_authoritative_doc(docs)["id"] == "newer"

    def test_supports_docstatus_enum_and_z_suffix(self):
        """兼容 DocStatus 枚举与 Z 结尾时间戳。"""
        docs = [
            {
                "id": "failed",
                "status": DocStatus.FAILED,
                "updated_at": "2026-01-04T00:00:00Z",
            },
            {
                "id": "processed",
                "status": DocStatus.PROCESSED,
                "updated_at": "2026-01-01T00:00:00Z",
            },
        ]
        assert select_authoritative_doc(docs)["id"] == "processed"

    def test_empty_returns_none(self):
        assert select_authoritative_doc([]) is None

    def test_missing_updated_at_still_selects_by_priority(self):
        """updated_at 缺失时不影响状态优先级判断。"""
        docs = [
            {"id": "failed", "status": "failed"},
            {"id": "processed", "status": "processed"},
        ]
        assert select_authoritative_doc(docs)["id"] == "processed"


# ──────────────────────────────────────────────
# JSON 存储后端：真实实例接线验证
# ──────────────────────────────────────────────


class _SimpleTokenizerImpl:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


async def _dummy_embedding(texts: list[str]) -> np.ndarray:
    return np.ones((len(texts), 8), dtype=float)


async def _dummy_llm(*args, **kwargs) -> str:
    return "ok"


def _deterministic_chunking(
    tokenizer,
    content: str,
    split_by_character,
    split_by_character_only: bool,
    chunk_overlap_token_size: int,
    chunk_token_size: int,
) -> list[dict]:
    return [{"tokens": 1, "content": content, "chunk_order_index": 0}]


async def _build_rag(tmp_path):
    from lightrag.lightrag import LightRAG

    rag = LightRAG(
        working_dir=str(tmp_path / "rag"),
        workspace=f"authority_{uuid4().hex[:8]}",
        llm_model_func=_dummy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8,
            max_token_size=8192,
            func=_dummy_embedding,
        ),
        tokenizer=Tokenizer("test-tokenizer", _SimpleTokenizerImpl()),
        chunking_func=_deterministic_chunking,
        max_parallel_insert=1,
    )
    await rag.initialize_storages()
    return rag


def _doc(status: str, updated_at: str, track_id: str) -> dict:
    return {
        "status": status,
        "content_summary": "summary",
        "content_length": 1,
        "chunks_count": 0,
        "chunks_list": [],
        "file_path": "same.txt",
        "track_id": track_id,
        "updated_at": updated_at,
        "created_at": updated_at,
    }


@pytest.mark.asyncio
async def test_json_backend_returns_authoritative_record(tmp_path):
    """JSON 后端 by_file_path 应返回 processed，而非先扫到的 failed。"""
    rag = await _build_rag(tmp_path)
    try:
        await rag.doc_status.upsert(
            {
                "doc-failed": _doc(
                    "failed", "2026-01-04T00:00:00+00:00", "track-failed"
                ),
                "doc-processing": _doc(
                    "processing", "2026-01-03T00:00:00+00:00", "track-processing"
                ),
                "doc-processed": _doc(
                    "processed", "2026-01-01T00:00:00+00:00", "track-processed"
                ),
            }
        )

        doc = await rag.doc_status.get_doc_by_file_path("same.txt")
        assert doc is not None
        assert doc["status"] == "processed"
        assert doc["track_id"] == "track-processed"

        doc_id = await rag.doc_status.get_doc_id_by_file_path("same.txt")
        assert doc_id == "doc-processed"
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_json_backend_same_tier_takes_latest(tmp_path):
    """无 processed 时，进行中记录取 updated_at 最新的一条。"""
    rag = await _build_rag(tmp_path)
    try:
        await rag.doc_status.upsert(
            {
                "doc-pending-old": _doc(
                    "pending", "2026-01-01T00:00:00+00:00", "track-old"
                ),
                "doc-processing-new": _doc(
                    "processing", "2026-01-02T00:00:00+00:00", "track-new"
                ),
                "doc-failed-newest": _doc(
                    "failed", "2026-01-03T00:00:00+00:00", "track-failed"
                ),
            }
        )

        doc = await rag.doc_status.get_doc_by_file_path("same.txt")
        assert doc is not None
        assert doc["track_id"] == "track-new"
        assert (
            await rag.doc_status.get_doc_id_by_file_path("same.txt")
            == "doc-processing-new"
        )
    finally:
        await rag.finalize_storages()
