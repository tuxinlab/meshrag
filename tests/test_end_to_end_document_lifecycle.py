"""端到端：真实 upload/delete 端点 + 真实 LightRAG 实例走完整文档生命周期。

覆盖 Java 中台实际调用序列：
  上传 → 后台解析 → 按 file_path 查询 → 同名重解析 → 同名改内容 →
  删除后重传 → force 强制解析 → failed 记录重传。

说明：
- 存储为真实的本机存储（JSON KV / NanoVectorDB / NetworkX），LLM 与 embedding 使用
  测试桩；实体抽取阶段用假实现跳过（避免依赖真实 LLM 的 JSON 输出），
  其余管线（入队、分块、状态流转、删除、判重）全部为真实代码路径。
"""

from io import BytesIO
from types import MethodType
from uuid import uuid4

import numpy as np
import pytest
from fastapi import BackgroundTasks
from starlette.datastructures import UploadFile

from lightrag.base import DocStatus
from lightrag.lightrag import LightRAG
from lightrag.utils import (
    EmbeddingFunc,
    Tokenizer,
    compute_mdhash_id,
    sanitize_text_for_encoding,
)

pytestmark = pytest.mark.offline


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
    return [
        {"tokens": 1, "content": f"{content}::chunk1", "chunk_order_index": 0},
        {"tokens": 1, "content": f"{content}::chunk2", "chunk_order_index": 1},
    ]


def _doc_id(text: str) -> str:
    return compute_mdhash_id(sanitize_text_for_encoding(text), prefix="doc-")


def _status_of(doc: dict) -> str:
    status = doc.get("status")
    return status.value if isinstance(status, DocStatus) else str(status)


async def _build_rag(tmp_path) -> LightRAG:
    workspace = f"e2e_{uuid4().hex[:8]}"
    rag = LightRAG(
        working_dir=str(tmp_path / "rag"),
        workspace=workspace,
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

    # 真实管线中仅替换“实体抽取”这一步，避免依赖真实 LLM 的 JSON 输出
    async def _fake_extract(self, chunks, pipeline_status, pipeline_status_lock):
        # 每个 chunk 返回一份空的 (entities, relations) 结果，跳过真实 LLM 抽取
        return [({}, {}) for _ in chunks]

    rag._process_extract_entities = MethodType(_fake_extract, rag)
    return rag


@pytest.fixture()
async def e2e(tmp_path):
    from lightrag.api.routers.document_routes import (
        DeleteDocRequest,
        create_document_routes,
    )
    from unittest.mock import AsyncMock, MagicMock

    rag = await _build_rag(tmp_path)

    mock_rag_mgr = MagicMock()
    mock_rag_mgr.get_rag = AsyncMock(return_value=rag)
    mock_doc_mgr = MagicMock()
    mock_doc_mgr.base_input_dir = str(tmp_path)

    router = create_document_routes(mock_rag_mgr, mock_doc_mgr, api_key=None)
    upload_route = [
        r for r in router.routes if getattr(r, "path", None) == "/documents/upload"
    ][-1]
    delete_route = [
        r
        for r in router.routes
        if getattr(r, "path", None) == "/documents/delete_document"
    ][-1]

    async def _run_background(background_tasks: BackgroundTasks):
        for task in background_tasks.tasks:
            await task.func(*task.args, **task.kwargs)

    async def upload(filename: str, content: bytes, force=False, overwrite=False):
        background_tasks = BackgroundTasks()
        file = UploadFile(file=BytesIO(content), filename=filename)
        resp = await upload_route.endpoint(
            kb_id=rag.workspace,
            background_tasks=background_tasks,
            file=file,
            overwrite=overwrite,
            force=force,
        )
        await _run_background(background_tasks)
        return resp

    async def delete(doc_id: str):
        background_tasks = BackgroundTasks()
        request = DeleteDocRequest(
            doc_ids=[doc_id], delete_file=True, delete_llm_cache=False
        )
        await delete_route.endpoint(
            kb_id=rag.workspace,
            delete_request=request,
            background_tasks=background_tasks,
        )
        await _run_background(background_tasks)

    yield rag, upload, delete
    await rag.finalize_storages()


@pytest.mark.asyncio
async def test_upload_then_same_content_reparse_returns_already_processed(e2e):
    """上传并解析完成后，同名同内容重解析应返回 already_processed。"""
    rag, upload, _ = e2e
    text = "alpha content"

    resp = await upload("a.txt", text.encode())
    assert resp.status == "success"

    doc = await rag.doc_status.get_by_id(_doc_id(text))
    assert doc is not None
    assert _status_of(doc) == "processed", "真实管线应完成解析"

    resp2 = await upload("a.txt", text.encode())
    assert resp2.status == "already_processed"
    assert resp2.doc_id == _doc_id(text)


@pytest.mark.asyncio
async def test_same_name_changed_content_is_rejected_without_losing_old(e2e):
    """同名但内容变化：返回 duplicated，旧内容保持不变，新内容不入队。"""
    rag, upload, _ = e2e
    old_text = "beta version one"
    new_text = "beta version two"

    resp = await upload("b.txt", old_text.encode())
    assert resp.status == "success"
    old_doc = await rag.doc_status.get_by_id(_doc_id(old_text))
    assert _status_of(old_doc) == "processed"

    resp2 = await upload("b.txt", new_text.encode())
    assert resp2.status == "duplicated"

    assert await rag.doc_status.get_by_id(_doc_id(new_text)) is None
    preserved = await rag.doc_status.get_by_id(_doc_id(old_text))
    assert preserved is not None and _status_of(preserved) == "processed"
    full_doc = await rag.full_docs.get_by_id(_doc_id(old_text))
    assert old_text in full_doc["content"]


@pytest.mark.asyncio
async def test_delete_then_reupload_succeeds(e2e):
    """删除后 by_file_path 应为空，重新上传可正常解析（不再永久提示重复）。"""
    rag, upload, delete = e2e
    text = "gamma content"
    doc_id = _doc_id(text)

    assert (await upload("c.txt", text.encode())).status == "success"
    assert _status_of(await rag.doc_status.get_by_id(doc_id)) == "processed"

    await delete(doc_id)
    assert await rag.doc_status.get_by_id(doc_id) is None
    assert await rag.doc_status.get_doc_by_file_path("c.txt") is None

    resp = await upload("c.txt", text.encode())
    assert resp.status == "success"
    assert _status_of(await rag.doc_status.get_by_id(doc_id)) == "processed"


@pytest.mark.asyncio
async def test_force_reparse_requeues_same_content(e2e):
    """force=true 跳过判重，重新入队并生成新的 track_id。"""
    rag, upload, _ = e2e
    text = "delta content"
    doc_id = _doc_id(text)

    assert (await upload("d.txt", text.encode())).status == "success"
    first = await rag.doc_status.get_by_id(doc_id)
    assert _status_of(first) == "processed"

    resp = await upload("d.txt", text.encode(), force=True)
    assert resp.status == "success"

    second = await rag.doc_status.get_by_id(doc_id)
    assert _status_of(second) == "processed"
    assert second.get("track_id") != first.get("track_id"), "force 应重新入队"


@pytest.mark.asyncio
async def test_failed_record_allows_reupload(e2e):
    """failed 记录被视为无效指纹，重新上传应放行并完成解析。"""
    rag, upload, _ = e2e
    text = "epsilon content"
    doc_id = _doc_id(text)

    await rag.doc_status.upsert(
        {
            doc_id: {
                "status": DocStatus.FAILED,
                "content_summary": "summary",
                "content_length": len(text),
                "chunks_count": 0,
                "chunks_list": [],
                "error_msg": "LLM unavailable",
                "file_path": "e.txt",
                "track_id": "old_track",
                "metadata": {},
            }
        }
    )

    resp = await upload("e.txt", text.encode())
    assert resp.status == "success"
    doc = await rag.doc_status.get_by_id(doc_id)
    assert _status_of(doc) == "processed"
