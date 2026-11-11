"""
GET /models 接口测试。

覆盖 lightrag/api/routers/models_routes.py：
- 已配置 embedding 模型时返回 configured
- 未配置时解析 provider 默认模型（provider_default）
- 无法解析默认模型时返回 null + unknown
- dim / max_token_size 取运行时 EmbeddingFunc 实际值
- 响应不泄露任何 api_key
"""

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lightrag.utils import EmbeddingFunc


async def _dummy_embed(texts):
    """测试用 embedding 函数。"""
    return [[0.0] * 8 for _ in texts]


def _make_args(**overrides):
    """构造测试用启动参数。"""
    base = dict(
        llm_binding="ollama",
        llm_binding_host="http://127.0.0.1:11434",
        llm_model="qwen3:8b",
        embedding_binding="ollama",
        embedding_binding_host="http://127.0.0.1:11434",
        embedding_model="qwen3-embedding:0.6b",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _make_embedding_func(
    embedding_dim=1024,
    max_token_size=8192,
    effective_model_name=None,
    model_source=None,
):
    """构造带运行期解析信息的 EmbeddingFunc。"""
    func = EmbeddingFunc(
        embedding_dim=embedding_dim,
        max_token_size=max_token_size,
        func=_dummy_embed,
    )
    if effective_model_name is not None:
        func.effective_model_name = effective_model_name
    if model_source is not None:
        func.model_source = model_source
    return func


def _make_client(args, embedding_func):
    from lightrag.api.routers.models_routes import create_models_routes

    app = FastAPI()
    app.include_router(create_models_routes(args, embedding_func, api_key=None))
    return TestClient(app)


@pytest.mark.offline
class TestModelsEndpoint:
    """验证 /models 的返回内容。"""

    def test_configured_embedding_model(self):
        """配置了 EMBEDDING_MODEL 时应返回 configured。"""
        args = _make_args()
        embedding_func = _make_embedding_func(
            effective_model_name="qwen3-embedding:0.6b",
            model_source="configured",
        )

        response = _make_client(args, embedding_func).get("/models")

        assert response.status_code == 200
        data = response.json()
        assert data["llm"] == {
            "binding": "ollama",
            "host": "http://127.0.0.1:11434",
            "model": "qwen3:8b",
        }
        assert data["embedding"] == {
            "binding": "ollama",
            "host": "http://127.0.0.1:11434",
            "model": "qwen3-embedding:0.6b",
            "configured_model": "qwen3-embedding:0.6b",
            "model_source": "configured",
            "dim": 1024,
            "max_token_size": 8192,
        }

    def test_unconfigured_uses_provider_default(self):
        """未配置 EMBEDDING_MODEL 时应返回 provider 默认模型。"""
        args = _make_args(embedding_model=None)
        embedding_func = _make_embedding_func(
            effective_model_name="bge-m3:latest",
            model_source="provider_default",
        )

        data = _make_client(args, embedding_func).get("/models").json()

        assert data["embedding"]["model"] == "bge-m3:latest"
        assert data["embedding"]["configured_model"] is None
        assert data["embedding"]["model_source"] == "provider_default"

    def test_unresolvable_model_returns_null(self):
        """既未配置也无法解析默认模型时，model 为 null 且 source 为 unknown。"""
        args = _make_args(embedding_model=None)
        embedding_func = _make_embedding_func()

        data = _make_client(args, embedding_func).get("/models").json()

        assert data["embedding"]["model"] is None
        assert data["embedding"]["model_source"] == "unknown"
        # 维度仍应来自运行时 EmbeddingFunc
        assert data["embedding"]["dim"] == 1024

    def test_dimension_comes_from_runtime_embedding_func(self):
        """dim / max_token_size 必须取运行时 EmbeddingFunc，而非 .env 原始值。"""
        args = _make_args()
        embedding_func = _make_embedding_func(
            embedding_dim=768,
            max_token_size=4096,
            effective_model_name="custom-model",
            model_source="configured",
        )

        data = _make_client(args, embedding_func).get("/models").json()

        assert data["embedding"]["dim"] == 768
        assert data["embedding"]["max_token_size"] == 4096

    def test_response_does_not_expose_api_keys(self):
        """响应中不得出现任何 api_key 字段或取值。"""
        args = _make_args()
        args.llm_binding_api_key = "sk-llm-secret"
        args.embedding_binding_api_key = "sk-embed-secret"
        embedding_func = _make_embedding_func(
            effective_model_name="qwen3-embedding:0.6b",
            model_source="configured",
        )

        body = _make_client(args, embedding_func).get("/models").text

        assert "sk-llm-secret" not in body
        assert "sk-embed-secret" not in body
        assert "api_key" not in body


@pytest.mark.offline
class TestProviderDefaultModel:
    """验证 provider 默认模型解析。"""

    def test_reads_model_default(self):
        """签名中的 model 字符串默认值应被解析。"""
        from lightrag.api.lightrag_server import (
            get_embedding_provider_default_model,
        )

        async def _provider(texts, model="text-embedding-default"):
            return [[0.0] for _ in texts]

        func = EmbeddingFunc(embedding_dim=8, func=_provider)

        assert get_embedding_provider_default_model(func) == "text-embedding-default"

    def test_reads_embed_model_default(self):
        """签名中的 embed_model 字符串默认值应被解析。"""
        from lightrag.api.lightrag_server import (
            get_embedding_provider_default_model,
        )

        async def _provider(texts, embed_model="bge-default"):
            return [[0.0] for _ in texts]

        func = EmbeddingFunc(embedding_dim=8, func=_provider)

        assert get_embedding_provider_default_model(func) == "bge-default"

    def test_returns_none_without_string_default(self):
        """没有字符串默认值（如 Azure 部署名）时应返回 None。"""
        from lightrag.api.lightrag_server import (
            get_embedding_provider_default_model,
        )

        async def _provider(texts, model=None):
            return [[0.0] for _ in texts]

        func = EmbeddingFunc(embedding_dim=8, func=_provider)

        assert get_embedding_provider_default_model(func) is None
