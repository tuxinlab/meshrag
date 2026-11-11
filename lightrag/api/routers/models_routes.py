"""
Runtime model information routes.

提供 ``GET /models``：返回服务当前实际使用的 LLM 与向量模型。
"""

from typing import Any, Optional

from fastapi import APIRouter, Depends

from lightrag.api.utils_api import get_combined_auth_dependency


def build_models_info(args, embedding_func) -> dict[str, Any]:
    """汇总当前运行实际使用的模型信息。

    向量模型优先取运行期解析结果（``EmbeddingFunc`` 上由
    ``create_optimized_embedding_function`` 写入的 ``effective_model_name``），
    其次是 ``EMBEDDING_MODEL`` 配置值；维度与 max_token_size 取
    ``EmbeddingFunc`` 的实际属性，而非 .env 原始值。

    Args:
        args: 服务启动参数（argparse.Namespace）。
        embedding_func: 运行期 EmbeddingFunc 实例。

    Returns:
        可直接序列化为 JSON 的模型信息字典。
    """
    configured_embedding_model = getattr(args, "embedding_model", None)
    runtime_embedding_model = getattr(embedding_func, "effective_model_name", None)
    effective_embedding_model = runtime_embedding_model or configured_embedding_model

    model_source = getattr(embedding_func, "model_source", None)
    if not model_source:
        model_source = "configured" if configured_embedding_model else "unknown"

    return {
        "llm": {
            "binding": getattr(args, "llm_binding", None),
            "host": getattr(args, "llm_binding_host", None),
            "model": getattr(args, "llm_model", None),
        },
        "embedding": {
            "binding": getattr(args, "embedding_binding", None),
            "host": getattr(args, "embedding_binding_host", None),
            "model": effective_embedding_model,
            "configured_model": configured_embedding_model,
            "model_source": model_source,
            "dim": getattr(embedding_func, "embedding_dim", None),
            "max_token_size": getattr(embedding_func, "max_token_size", None),
        },
    }


def create_models_routes(
    args, embedding_func, api_key: Optional[str] = None
) -> APIRouter:
    """创建模型信息路由。

    Args:
        args: 服务启动参数。
        embedding_func: 运行期 EmbeddingFunc 实例。
        api_key: 可选的 API Key，用于鉴权。

    Returns:
        已注册 ``GET /models`` 的 APIRouter。
    """
    router = APIRouter(tags=["models"])
    combined_auth = get_combined_auth_dependency(api_key)

    @router.get(
        "/models",
        dependencies=[Depends(combined_auth)],
        summary="Get the LLM and embedding models currently in use",
        description=(
            "Returns the LLM binding/model/host and the embedding "
            "binding/model/host actually used by the running service, "
            "including the resolved embedding dimension and max token size."
        ),
    )
    async def get_models():
        return build_models_info(args, embedding_func)

    return router
