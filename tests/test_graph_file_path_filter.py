"""
``/graphs`` 按单个文档筛选知识图谱的测试。

覆盖：
- NetworkXStorage.get_knowledge_graph(file_path=...) 的精确过滤与截断计数
- LightRAG.get_knowledge_graph 对不支持预过滤存储的后置过滤降级
- ``GET /graphs`` 的 file_path 透传与 file_paths 全量返回
"""

import asyncio
from types import SimpleNamespace

import networkx as nx
import pytest

from lightrag.base import DocStatus
from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.kg.networkx_impl import NetworkXStorage
from lightrag.types import (
    KnowledgeGraph,
    KnowledgeGraphEdge,
    KnowledgeGraphNode,
)

FILE_A = "a.pdf"
FILE_B = "b.pdf"
FILE_C = "c.pdf"


def _node_props(*file_paths: str) -> dict:
    """构造节点属性，file_path 使用图存储约定的分隔符拼接。"""
    return {
        "file_path": GRAPH_FIELD_SEP.join(file_paths),
        "entity_type": "Person",
        "description": "test entity",
    }


def _build_graph() -> nx.Graph:
    """构造 3 个文件的小图，含 1 个 A/B 共享实体和 1 个文件名子串陷阱节点。"""
    graph = nx.Graph()
    graph.add_node("Alice", **_node_props(FILE_A))
    graph.add_node("Bob", **_node_props(FILE_B))
    graph.add_node("Carol", **_node_props(FILE_C))
    # 共享实体：同时出现在 a.pdf 与 b.pdf
    graph.add_node("Shared", **_node_props(FILE_A, FILE_B))
    # 子串陷阱：xxxa.pdf 不应被 a.pdf 命中
    graph.add_node("OtherA", **_node_props("xxx" + FILE_A))

    # 边属性故意写成 b.pdf，用于验证边不做二次 file_path 过滤
    graph.add_edge("Alice", "Shared", file_path=FILE_B, description="a-shared")
    graph.add_edge("Shared", "Bob", file_path=FILE_B, description="shared-b")
    graph.add_edge("Bob", "Carol", file_path=FILE_C, description="b-c")
    return graph


def _make_storage(tmp_path: str, graph: nx.Graph) -> NetworkXStorage:
    """绕过 shared_storage 初始化，构造可直接查询的 NetworkXStorage。"""
    storage = NetworkXStorage(
        namespace="graph_chunk_entity_relation",
        workspace="kb_test",
        global_config={"working_dir": tmp_path, "max_graph_nodes": 1000},
        embedding_func=None,
    )
    storage._graph = graph
    storage._storage_lock = asyncio.Lock()
    storage.storage_updated = SimpleNamespace(value=False)
    return storage


# ──────────────────────────────────────────────
# NetworkXStorage 层
# ──────────────────────────────────────────────


@pytest.fixture()
def nx_storage(tmp_path):
    return _make_storage(str(tmp_path), _build_graph())


@pytest.mark.offline
def test_networkx_declares_file_path_filter_support():
    """NetworkXStorage 需声明支持 file_path 预过滤。"""
    assert NetworkXStorage.supports_file_path_filter is True


@pytest.mark.offline
async def test_filter_keeps_file_nodes_and_shared_entity(nx_storage):
    """按 a.pdf 过滤：保留 A 的实体与 A/B 共享实体，边只保留两端都在集合内的。"""
    result = await nx_storage.get_knowledge_graph(
        "*", max_depth=3, max_nodes=100, file_path=FILE_A
    )

    assert {node.id for node in result.nodes} == {"Alice", "Shared"}
    # Alice-Shared 的边属性写的是 b.pdf，说明边未被二次过滤
    assert {edge.id for edge in result.edges} == {"Alice-Shared"}


@pytest.mark.offline
async def test_filter_uses_exact_separator_match(nx_storage):
    """file_path 必须按分隔符精确匹配，子串同名文件不得命中。"""
    result = await nx_storage.get_knowledge_graph(
        "*", max_depth=3, max_nodes=100, file_path=FILE_A
    )

    assert "OtherA" not in {node.id for node in result.nodes}


@pytest.mark.offline
async def test_truncation_is_counted_after_filtering(nx_storage):
    """max_nodes 小于过滤后节点数时 is_truncated=True，且计数基于过滤后集合。"""
    result = await nx_storage.get_knowledge_graph(
        "*", max_depth=3, max_nodes=1, file_path=FILE_A
    )

    assert result.is_truncated is True
    assert len(result.nodes) == 1


@pytest.mark.offline
@pytest.mark.parametrize("file_path", [None, ""])
async def test_no_file_path_returns_whole_graph(nx_storage, file_path):
    """file_path 为空或 None 时与改动前行为一致（整库图）。"""
    result = await nx_storage.get_knowledge_graph(
        "*", max_depth=3, max_nodes=100, file_path=file_path
    )

    assert {node.id for node in result.nodes} == {
        "Alice",
        "Bob",
        "Carol",
        "Shared",
        "OtherA",
    }
    assert len(result.edges) == 3
    assert result.is_truncated is False


# ──────────────────────────────────────────────
# LightRAG 层：后置过滤降级
# ──────────────────────────────────────────────


def _knowledge_graph() -> KnowledgeGraph:
    return KnowledgeGraph(
        nodes=[
            KnowledgeGraphNode(
                id="Alice", labels=["Alice"], properties={"file_path": FILE_A}
            ),
            KnowledgeGraphNode(
                id="Shared",
                labels=["Shared"],
                properties={"file_path": f"{FILE_A}{GRAPH_FIELD_SEP}{FILE_B}"},
            ),
            KnowledgeGraphNode(
                id="Bob", labels=["Bob"], properties={"file_path": FILE_B}
            ),
            KnowledgeGraphNode(
                id="OtherA",
                labels=["OtherA"],
                properties={"file_path": "xxx" + FILE_A},
            ),
        ],
        edges=[
            KnowledgeGraphEdge(
                id="Alice-Shared",
                type="DIRECTED",
                source="Alice",
                target="Shared",
                properties={},
            ),
            KnowledgeGraphEdge(
                id="Shared-Bob",
                type="DIRECTED",
                source="Shared",
                target="Bob",
                properties={},
            ),
        ],
        is_truncated=True,
    )


@pytest.mark.offline
def test_post_filter_graph_by_file_path():
    """后置过滤函数：节点精确匹配，边要求两端都被保留。"""
    from lightrag.lightrag import _filter_graph_by_file_path

    filtered = _filter_graph_by_file_path(_knowledge_graph(), FILE_A)

    assert {node.id for node in filtered.nodes} == {"Alice", "Shared"}
    assert {edge.id for edge in filtered.edges} == {"Alice-Shared"}
    # 截断标记沿用存储层结果
    assert filtered.is_truncated is True


@pytest.mark.offline
async def test_lightrag_post_filters_for_unsupporting_storage():
    """存储不支持预过滤时，LightRAG 先取图再后置过滤。"""
    from lightrag.lightrag import LightRAG

    class _UnsupportingStorage:
        async def get_knowledge_graph(self, node_label, max_depth, max_nodes):
            return _knowledge_graph()

    rag = object.__new__(LightRAG)
    rag.chunk_entity_relation_graph = _UnsupportingStorage()
    rag.max_graph_nodes = 1000
    rag.workspace = "kb_test"

    result = await rag.get_knowledge_graph("*", 3, 100, file_path=FILE_A)

    assert {node.id for node in result.nodes} == {"Alice", "Shared"}
    assert {edge.id for edge in result.edges} == {"Alice-Shared"}


@pytest.mark.offline
@pytest.mark.parametrize("file_path", [None, ""])
async def test_lightrag_unsupporting_storage_without_file_path(file_path):
    """存储不支持预过滤且未传 file_path 时，不得向存储传 file_path 关键字。

    回归：Neo4JStorage 等方法签名只有 3 个参数，传 file_path=None 会 TypeError。
    """
    from lightrag.lightrag import LightRAG

    class _UnsupportingStorage:
        async def get_knowledge_graph(self, node_label, max_depth, max_nodes):
            return _knowledge_graph()

    rag = object.__new__(LightRAG)
    rag.chunk_entity_relation_graph = _UnsupportingStorage()
    rag.max_graph_nodes = 1000
    rag.workspace = "kb_test"

    result = await rag.get_knowledge_graph("*", 3, 100, file_path=file_path)

    # 整库返回，不做任何过滤
    assert len(result.nodes) == 4
    assert len(result.edges) == 2


@pytest.mark.offline
async def test_lightrag_delegates_file_path_to_supporting_storage():
    """存储支持预过滤时，file_path 直接透传给存储层。"""
    from lightrag.lightrag import LightRAG

    class _SupportingStorage:
        supports_file_path_filter = True

        def __init__(self):
            self.received = []

        async def get_knowledge_graph(
            self, node_label, max_depth, max_nodes, file_path=None
        ):
            self.received.append(file_path)
            return KnowledgeGraph()

    storage = _SupportingStorage()
    rag = object.__new__(LightRAG)
    rag.chunk_entity_relation_graph = storage
    rag.max_graph_nodes = 1000
    rag.workspace = "kb_test"

    await rag.get_knowledge_graph("*", 3, 100, file_path=FILE_A)

    assert storage.received == [FILE_A]


# ──────────────────────────────────────────────
# 路由层：file_path 透传 + file_paths 全量返回
# ──────────────────────────────────────────────


class _FakeDoc:
    def __init__(self, file_path):
        self.file_path = file_path


class _FakeDocStatus:
    async def get_docs_by_status(self, status):
        assert status == DocStatus.PROCESSED
        return {
            "d1": _FakeDoc(FILE_B),
            "d2": _FakeDoc(FILE_A),
            "d3": _FakeDoc(FILE_A),  # 重复文件去重
            "d4": _FakeDoc(None),  # 无文件名记录应忽略
        }


class _FakeRag:
    def __init__(self):
        self.doc_status = _FakeDocStatus()
        self.calls = []

    async def get_knowledge_graph(
        self, node_label, max_depth, max_nodes, file_path=None
    ):
        self.calls.append(file_path)
        return KnowledgeGraph()


class _FakeRagManager:
    def __init__(self, rag):
        self.rag = rag

    async def get_rag(self, kb_id):
        return self.rag

    async def get_rag_or_none(self, kb_id):
        return self.rag


@pytest.fixture(scope="module")
def fake_rag():
    return _FakeRag()


@pytest.fixture(scope="module")
def graphs_endpoint(fake_rag):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from lightrag.api.routers.graph_routes import create_graph_routes

    # MeshRAG 的 create_graph_routes 每次调用返回独立 router，无模块级共享状态
    kb_router = create_graph_routes(_FakeRagManager(fake_rag))

    endpoint = next(
        (
            route.endpoint
            for route in kb_router.routes
            if getattr(route, "path", None) == "/graphs"
        ),
        None,
    )
    if endpoint is None:
        raise AssertionError("/graphs endpoint not registered")

    # 用返回的独立 router 构建独立 app，验证 FastAPI 参数解析与响应序列化
    app = FastAPI()
    app.include_router(kb_router)

    with TestClient(app) as client:
        yield endpoint, client


@pytest.mark.offline
async def test_graphs_returns_all_processed_file_paths(graphs_endpoint, fake_rag):
    """未传 file_path 时返回整库文件列表，且透传 None。"""
    endpoint, _ = graphs_endpoint
    fake_rag.calls.clear()

    # 直接调用端点函数，需显式传入 Query 的默认值（绕过 FastAPI 参数解析）
    result = await endpoint(
        kb_id="kb1", label="*", max_depth=3, max_nodes=1000, file_path=None
    )

    assert result.file_paths == [FILE_A, FILE_B]
    assert fake_rag.calls == [None]


@pytest.mark.offline
async def test_graphs_passes_file_path_and_still_returns_all_file_paths(
    graphs_endpoint, fake_rag
):
    """传 file_path 时透传该值，file_paths 仍为全量列表。"""
    endpoint, _ = graphs_endpoint
    fake_rag.calls.clear()

    result = await endpoint(
        kb_id="kb1", label="*", max_depth=3, max_nodes=1000, file_path=FILE_A
    )

    assert fake_rag.calls == [FILE_A]
    assert result.file_paths == [FILE_A, FILE_B]


@pytest.mark.offline
async def test_graphs_treats_empty_file_path_as_whole_graph(graphs_endpoint, fake_rag):
    """空串 file_path 按 None 处理，避免把全库图过滤成空图。"""
    endpoint, _ = graphs_endpoint
    fake_rag.calls.clear()

    await endpoint(kb_id="kb1", label="*", max_depth=3, max_nodes=1000, file_path="")

    assert fake_rag.calls == [None]


@pytest.mark.offline
def test_graphs_http_serializes_file_paths(graphs_endpoint, fake_rag):
    """HTTP 层：默认参数解析正确，响应对外序列化 file_paths。"""
    _, client = graphs_endpoint
    fake_rag.calls.clear()

    response = client.get("/graphs", params={"kb_id": "kb1", "label": "*"})
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["file_paths"] == [FILE_A, FILE_B]
    assert body["nodes"] == []
    assert body["edges"] == []
    assert body["is_truncated"] is False
    assert fake_rag.calls == [None]

    # 空串经 FastAPI 解析后仍按整库处理
    response = client.get(
        "/graphs", params={"kb_id": "kb1", "label": "*", "file_path": ""}
    )
    assert response.status_code == 200, response.text
    assert fake_rag.calls[-1] is None
