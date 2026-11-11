"""文档上传判重状态改造测试。

覆盖契约：
- 文件名闸：processed → already_processed；进行中 → processing；failed → 放行
- 内容闸：同名记录区分“自己”；不同名同内容仍 → duplicated；failed 放行
- force=true：跳过判重，并以 overwrite=True 重新入队
"""

from io import BytesIO
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import BackgroundTasks
from starlette.datastructures import UploadFile

pytestmark = pytest.mark.offline


# ──────────────────────────────────────────────
# 内容闸：_check_content_duplicate 区分“自己”
# ──────────────────────────────────────────────


class TestContentCheckSelf:
    """同名记录属于本次上传自身，应按状态返回对应结果。"""

    @staticmethod
    def _prepare(tmp_path, text: str, filename: str, existing_file_path: str):
        from lightrag.utils import compute_mdhash_id, sanitize_text_for_encoding

        p = tmp_path / filename
        p.write_text(text, encoding="utf-8")
        doc_id = compute_mdhash_id(sanitize_text_for_encoding(text), prefix="doc-")
        existing = {
            "id": doc_id,
            "track_id": "old_track",
            "file_path": existing_file_path,
        }
        mock_rag = MagicMock()
        mock_rag.doc_status = AsyncMock()
        mock_rag.doc_status.get_by_id = AsyncMock(return_value=existing)
        return p, doc_id, existing, mock_rag

    @pytest.mark.asyncio
    async def test_same_name_processed_returns_already_processed(self, tmp_path):
        """同名且已处理完成 → already_processed，并清理临时文件。"""
        from lightrag.api.routers.document_routes import _check_content_duplicate

        p, doc_id, existing, mock_rag = self._prepare(
            tmp_path, "same name processed", "doc.txt", "doc.txt"
        )
        existing["status"] = "processed"

        resp = await _check_content_duplicate(mock_rag, p)

        assert resp is not None
        assert resp.status == "already_processed"
        assert resp.doc_id == doc_id
        assert resp.track_id == "old_track"
        assert resp.original_file_path == "doc.txt"
        assert not p.exists(), "命中同名已处理内容时应清理临时文件"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", ["pending", "processing", "preprocessed"])
    async def test_same_name_in_progress_returns_processing(self, tmp_path, status):
        """同名且正在处理中 → processing，并清理临时文件。"""
        from lightrag.api.routers.document_routes import _check_content_duplicate

        p, doc_id, existing, mock_rag = self._prepare(
            tmp_path, f"same name {status}", "doc.txt", "doc.txt"
        )
        existing["status"] = status

        resp = await _check_content_duplicate(mock_rag, p)

        assert resp is not None
        assert resp.status == "processing"
        assert resp.doc_id == doc_id
        assert resp.track_id == "old_track"
        assert resp.original_file_path is None
        assert not p.exists(), "命中同名处理中内容时应清理临时文件"

    @pytest.mark.asyncio
    async def test_same_name_failed_returns_none(self, tmp_path):
        """同名且解析失败 → 放行，保留临时文件交入队。"""
        from lightrag.api.routers.document_routes import _check_content_duplicate

        p, _, existing, mock_rag = self._prepare(
            tmp_path, "same name failed", "doc.txt", "doc.txt"
        )
        existing["status"] = "failed"

        assert await _check_content_duplicate(mock_rag, p) is None
        assert p.exists(), "FAILED 放行时不应删除临时文件"

    @pytest.mark.asyncio
    async def test_different_name_same_content_returns_duplicated(self, tmp_path):
        """文件名不同但内容相同 → duplicated（维持现状）。"""
        from lightrag.api.routers.document_routes import _check_content_duplicate

        p, doc_id, existing, mock_rag = self._prepare(
            tmp_path, "different name same content", "doc.txt", "other.pdf"
        )
        existing["status"] = "processed"

        resp = await _check_content_duplicate(mock_rag, p)

        assert resp is not None
        assert resp.status == "duplicated"
        assert resp.doc_id == doc_id
        assert resp.original_file_path == "other.pdf"
        assert not p.exists()

    @pytest.mark.asyncio
    async def test_same_name_unreadable_content_returns_duplicated(self, tmp_path):
        """同名记录存在但内容无法提取时 → duplicated（无法证明内容未变化）。"""
        from lightrag.api.routers.document_routes import _check_content_duplicate

        p = tmp_path / "doc.unknownext"
        p.write_bytes(b"binary")
        mock_rag = MagicMock()
        mock_rag.doc_status = AsyncMock()
        mock_rag.doc_status.get_by_id = AsyncMock(return_value=None)

        same_name_existing = {
            "status": "processed",
            "track_id": "old_track",
            "file_path": "doc.unknownext",
            "doc_id": "doc-old",
        }

        resp = await _check_content_duplicate(
            mock_rag, p, same_name_existing=same_name_existing
        )

        assert resp is not None
        assert resp.status == "duplicated"
        assert resp.doc_id == "doc-old"
        assert not p.exists(), "无法判定内容时应清理临时文件"

    @pytest.mark.asyncio
    async def test_same_name_changed_content_returns_duplicated(self, tmp_path):
        """同名但内容已变化 → duplicated，并清理临时文件。

        内容 hash 未命中（get_by_id 为 None），但文件名闸已发现同名
        processed / 进行中记录，说明是“同名覆盖上传”，不能判为已处理。
        """
        from lightrag.api.routers.document_routes import _check_content_duplicate

        p = tmp_path / "doc.txt"
        p.write_text("new content", encoding="utf-8")
        mock_rag = MagicMock()
        mock_rag.doc_status = AsyncMock()
        mock_rag.doc_status.get_by_id = AsyncMock(return_value=None)

        same_name_existing = {
            "status": "processed",
            "track_id": "old_track",
            "file_path": "doc.txt",
            "doc_id": "doc-old",
        }

        resp = await _check_content_duplicate(
            mock_rag, p, same_name_existing=same_name_existing
        )

        assert resp is not None
        assert resp.status == "duplicated"
        assert resp.doc_id == "doc-old"
        assert resp.track_id == "old_track"
        assert resp.original_file_path == "doc.txt"
        assert not p.exists(), "同名内容变化时应清理临时文件"


# ──────────────────────────────────────────────
# 文件名闸 + force 入队
# ──────────────────────────────────────────────


@pytest.fixture()
def upload_endpoint(tmp_path):
    """构造 mock RAGManager 的 upload 端点：返回 (调用函数, mock_rag)。"""
    from lightrag.api.routers.document_routes import create_document_routes

    mock_rag = MagicMock()
    mock_rag.doc_status = MagicMock()
    mock_rag.doc_status.get_doc_by_file_path = AsyncMock(return_value=None)
    mock_rag.doc_status.get_doc_id_by_file_path = AsyncMock(return_value="doc-x")
    # 内容预检默认放行（内容不重复）
    mock_rag.doc_status.get_by_id = AsyncMock(return_value=None)
    mock_rag.adelete_by_doc_id = AsyncMock()

    mock_rag_mgr = MagicMock()
    mock_rag_mgr.get_rag = AsyncMock(return_value=mock_rag)

    mock_doc_mgr = MagicMock()
    mock_doc_mgr.base_input_dir = str(tmp_path)

    router = create_document_routes(mock_rag_mgr, mock_doc_mgr, api_key=None)
    # create_document_routes 向模块级 router 累积注册，取本次最新注册的一条
    route = [
        r for r in router.routes if getattr(r, "path", None) == "/documents/upload"
    ][-1]

    async def _call(
        filename: str = "doc.txt",
        content: bytes = b"hello",
        force=False,
        overwrite=False,
    ):
        background_tasks = BackgroundTasks()
        file = UploadFile(file=BytesIO(content), filename=filename)
        resp = await route.endpoint(
            kb_id="kb1",
            background_tasks=background_tasks,
            file=file,
            overwrite=overwrite,
            force=force,
        )
        return resp, background_tasks

    return _call, mock_rag


class TestUploadFilenameGate:
    """文件名闸按已有文档状态分流。"""

    @pytest.mark.asyncio
    async def test_processed_returns_already_processed(self, upload_endpoint):
        """同名且内容也相同 → already_processed（内容预检确认后返回）。"""
        from lightrag.utils import compute_mdhash_id, sanitize_text_for_encoding

        call_upload, mock_rag = upload_endpoint
        content = b"hello"
        doc_id = compute_mdhash_id(
            sanitize_text_for_encoding(content.decode()), prefix="doc-"
        )
        existing = {
            "id": doc_id,
            "status": "processed",
            "track_id": "old_track",
            "file_path": "doc.txt",
        }
        mock_rag.doc_status.get_doc_by_file_path = AsyncMock(return_value=existing)
        mock_rag.doc_status.get_doc_id_by_file_path = AsyncMock(return_value=doc_id)
        mock_rag.doc_status.get_by_id = AsyncMock(return_value=existing)

        resp, _ = await call_upload(content=content)

        assert resp.status == "already_processed"
        assert resp.doc_id == doc_id
        assert resp.track_id == "old_track"
        assert resp.original_file_path == "doc.txt"
        mock_rag.adelete_by_doc_id.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", ["pending", "processing", "preprocessed"])
    async def test_in_progress_returns_processing(self, upload_endpoint, status):
        """同名且内容也相同 → processing，并带 doc_id/track_id。"""
        from lightrag.utils import compute_mdhash_id, sanitize_text_for_encoding

        call_upload, mock_rag = upload_endpoint
        content = b"hello"
        doc_id = compute_mdhash_id(
            sanitize_text_for_encoding(content.decode()), prefix="doc-"
        )
        existing = {
            "id": doc_id,
            "status": status,
            "track_id": "old_track",
            "file_path": "doc.txt",
        }
        mock_rag.doc_status.get_doc_by_file_path = AsyncMock(return_value=existing)
        mock_rag.doc_status.get_doc_id_by_file_path = AsyncMock(return_value=doc_id)
        mock_rag.doc_status.get_by_id = AsyncMock(return_value=existing)

        resp, _ = await call_upload(content=content)

        assert resp.status == "processing"
        assert resp.doc_id == doc_id
        assert resp.track_id == "old_track"
        assert resp.original_file_path is None
        mock_rag.adelete_by_doc_id.assert_not_called()

    @pytest.mark.asyncio
    async def test_processed_same_name_changed_content_returns_duplicated(
        self, upload_endpoint, tmp_path
    ):
        """同名 processed 但内容已变化 → duplicated，且不误报已解析完成。"""
        call_upload, mock_rag = upload_endpoint
        mock_rag.doc_status.get_doc_by_file_path = AsyncMock(
            return_value={
                "id": "doc-old",
                "status": "processed",
                "track_id": "old_track",
                "file_path": "doc.txt",
            }
        )
        # 新内容 hash 未命中任何 doc_status 记录
        mock_rag.doc_status.get_by_id = AsyncMock(return_value=None)

        resp, background_tasks = await call_upload(content=b"updated content")

        assert resp.status == "duplicated", "同名内容变化不能被判为已处理"
        assert resp.doc_id == "doc-x"
        assert resp.track_id == "old_track"
        assert resp.original_file_path == "doc.txt"
        mock_rag.doc_status.get_by_id.assert_awaited(), "内容预检应被调用"
        assert not background_tasks.tasks, "同名内容变化不应入队"
        assert not (tmp_path / "kb1" / "doc.txt").exists(), "临时文件应被清理"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", ["pending", "processing", "preprocessed"])
    async def test_in_progress_same_name_changed_content_returns_duplicated(
        self, upload_endpoint, tmp_path, status
    ):
        """同名进行中但内容已变化 → duplicated（与改造前一致）。"""
        call_upload, mock_rag = upload_endpoint
        mock_rag.doc_status.get_doc_by_file_path = AsyncMock(
            return_value={
                "id": "doc-old",
                "status": status,
                "track_id": "old_track",
                "file_path": "doc.txt",
            }
        )
        mock_rag.doc_status.get_by_id = AsyncMock(return_value=None)

        resp, background_tasks = await call_upload(content=b"updated content")

        assert resp.status == "duplicated"
        mock_rag.doc_status.get_by_id.assert_awaited(), "内容预检应被调用"
        assert not background_tasks.tasks
        assert not (tmp_path / "kb1" / "doc.txt").exists()

    @pytest.mark.asyncio
    async def test_failed_allows_reupload(self, upload_endpoint):
        call_upload, mock_rag = upload_endpoint
        mock_rag.doc_status.get_doc_by_file_path = AsyncMock(
            return_value={
                "id": "doc-y",
                "status": "failed",
                "track_id": "old_track",
                "file_path": "doc.txt",
            }
        )

        resp, _ = await call_upload()

        assert resp.status == "success"
        mock_rag.adelete_by_doc_id.assert_not_called()


class TestForceReenqueue:
    """force=true 跳过判重并以 overwrite=True 重新入队。"""

    @pytest.mark.asyncio
    async def test_force_skips_checks_and_enqueues_with_overwrite(
        self, upload_endpoint
    ):
        call_upload, mock_rag = upload_endpoint
        # 同名 processed + 同内容 processed：正常情况下会被两道闸拦截
        mock_rag.doc_status.get_doc_by_file_path = AsyncMock(
            return_value={
                "id": "doc-y",
                "status": "processed",
                "track_id": "old_track",
                "file_path": "doc.txt",
            }
        )
        mock_rag.doc_status.get_by_id = AsyncMock(
            return_value={
                "id": "doc-content",
                "status": "processed",
                "track_id": "old_track",
                "file_path": "doc.txt",
            }
        )

        resp, background_tasks = await call_upload(force=True)

        assert resp.status == "success", "force 应跳过判重直接入队"
        # 跳过文件名闸与内容闸
        mock_rag.adelete_by_doc_id.assert_not_called()
        mock_rag.doc_status.get_by_id.assert_not_called()
        # 入队参数应带 overwrite=True，复用“删旧再入队”逻辑
        assert background_tasks.tasks, "force 应提交后台入队任务"
        task = background_tasks.tasks[-1]
        assert task.func.__name__ == "pipeline_index_file"
        assert task.kwargs.get("overwrite") is True


class TestOverwriteDeferredDeletion:
    """overwrite=true 的同名旧文档删除必须延后到同步校验通过之后。"""

    @pytest.mark.asyncio
    async def test_overwrite_skips_content_check_and_enqueues(self, upload_endpoint):
        """overwrite 跳过内容预检，校验通过后删除同名旧文档并以 overwrite 入队。

        回归点：改造前先删旧文档再跑内容预检，命中内容重复时返回 duplicated，
        旧文档已被删且新文档未入队（数据丢失）。
        """
        call_upload, mock_rag = upload_endpoint
        mock_rag.doc_status.get_doc_by_file_path = AsyncMock(
            return_value={
                "id": "doc-old",
                "status": "processed",
                "track_id": "old_track",
                "file_path": "doc.txt",
            }
        )
        mock_rag.doc_status.get_doc_id_by_file_path = AsyncMock(return_value="doc-old")
        # 内容命中另一文件，改造前会导致“先删后判重复”
        mock_rag.doc_status.get_by_id = AsyncMock(
            return_value={
                "id": "doc-other",
                "status": "processed",
                "track_id": "other_track",
                "file_path": "other.txt",
            }
        )

        resp, background_tasks = await call_upload(
            content=b"new content", overwrite=True
        )

        assert resp.status == "success", "overwrite 不应因内容预检返回 duplicated"
        assert not mock_rag.doc_status.get_by_id.await_count, "overwrite 应跳过内容预检"
        mock_rag.adelete_by_doc_id.assert_awaited_once_with("doc-old")
        assert background_tasks.tasks, "overwrite 应提交后台入队任务"
        task = background_tasks.tasks[-1]
        assert task.kwargs.get("overwrite") is True

    @pytest.mark.asyncio
    async def test_overwrite_does_not_delete_before_validation(
        self, upload_endpoint, tmp_path, monkeypatch
    ):
        """落盘前的孤儿清理失败时必须先返回，不得删除同名旧文档。"""
        call_upload, mock_rag = upload_endpoint
        mock_rag.doc_status.get_doc_by_file_path = AsyncMock(
            return_value={
                "id": "doc-old",
                "status": "processed",
                "track_id": "old_track",
                "file_path": "doc.txt",
            }
        )

        # 制造同名孤儿文件，并让 unlink 失败，触发布局前的 duplicated 返回
        orphan = tmp_path / "kb1" / "doc.txt"
        orphan.parent.mkdir(parents=True, exist_ok=True)
        orphan.write_text("stale", encoding="utf-8")

        from pathlib import Path

        real_unlink = Path.unlink

        def _fail_unlink(self, *args, **kwargs):
            if self == orphan:
                raise OSError("permission denied")
            return real_unlink(self, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", _fail_unlink)

        resp, background_tasks = await call_upload(
            content=b"new content", overwrite=True
        )

        assert resp.status == "duplicated", "孤儿清理失败应按原逻辑返回 duplicated"
        assert not mock_rag.adelete_by_doc_id.await_count, "校验失败时不得删除旧文档"
        assert not background_tasks.tasks
