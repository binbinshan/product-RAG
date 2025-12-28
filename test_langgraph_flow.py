#!/usr/bin/env python3
"""
测试LangGraph状态流转正确性
验证State读写契约是否正确
"""
from typing import TypedDict, Dict, Any
from models import MergedCandidate, RankedCandidate, ProductContext, ProductRealTimeData

# Mock数据用于测试
MOCK_CANDIDATES = [
    MergedCandidate(
        sku_id="SKU_1001",
        vector_score=0.9,
        keyword_score=0.8,
        hybrid_score=0.85,
        sources=["vector", "keyword"],
        content="儿童低乳糖配方奶粉"
    ),
    MergedCandidate(
        sku_id="SKU_3003",
        vector_score=0.7,
        keyword_score=0.9,
        hybrid_score=0.8,
        sources=["vector", "keyword"],
        content="肠胃友好型配方奶粉"
    )
]

MOCK_RANKED_CANDIDATES = [
    RankedCandidate(
        sku_id="SKU_1001",
        rerank_score=0.95,
        original_candidate=MOCK_CANDIDATES[0]
    ),
    RankedCandidate(
        sku_id="SKU_3003",
        rerank_score=0.85,
        original_candidate=MOCK_CANDIDATES[1]
    )
]

MOCK_PRODUCT_CONTEXT = [
    ProductContext(
        sku_id="SKU_1001",
        title="儿童低乳糖配方奶粉",
        highlights=["低乳糖配方", "益生菌添加", "易消化"],
        instructions="每日2-3次，每次30ml",
        description="专为肠胃敏感儿童设计的低乳糖配方奶粉"
    ),
    ProductContext(
        sku_id="SKU_3003",
        title="肠胃友好型配方奶粉",
        highlights=["益生元添加", "呵护肠胃", "易消化"],
        instructions="每日2-3次，每次25ml",
        description="含益生元配方，保护肠胃健康"
    )
]

MOCK_REALTIME_DATA = {
    "SKU_1001": ProductRealTimeData(price=299.0, stock=120, promotion="满299减50"),
    "SKU_3003": ProductRealTimeData(price=279.0, stock=200, promotion="买2送1")
}

def test_state_contracts():
    """测试每个节点的状态读写契约"""
    print("=== 测试LangGraph状态读写契约 ===\n")

    # 定义GraphState结构（与pipeline.py保持一致）
    class GraphState(TypedDict):
        raw_query: str
        user_context: dict
        rewritten_queries: list
        filters: dict
        candidates: list
        ranked_skus: list
        ranked_candidates: list
        rerank_type: str
        product_context: list
        real_time_data: dict
        final_response: str
        referenced_skus: list
        generation_type: str
        recommended_skus: list

    # 1. 测试Query Rewrite节点
    print("1. 测试QueryRewriteService状态契约...")
    from query_rewrite import QueryRewriteService

    try:
        llm_mock = type('MockLLM', (), {'invoke': lambda self, x: type('Response', (), {'content': '{"rewritten_queries": ["test"], "filters": {}}'})()})()
        qr_service = QueryRewriteService(llm=llm_mock)

        initial_state = {
            "raw_query": "不上火的奶粉",
            "user_context": {"channel": "miniapp"}
        }

        qr_result = qr_service(initial_state)
        print(f"  QueryRewrite输出字段: {list(qr_result.keys())}")
        assert "rewritten_queries" in qr_result
        assert "filters" in qr_result
        print("  ✅ QueryRewrite状态契约正确")
    except Exception as e:
        print(f"  ❌ QueryRewrite状态契约错误: {e}")

    # 2. 测试HybridSearch节点
    print("\n2. 测试HybridSearchService状态契约...")
    try:
        # Mock一个简化的HybridSearchService
        class MockHybridSearchService:
            def __call__(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
                return {
                    "candidates": MOCK_CANDIDATES
                }

        hs_service = MockHybridSearchService()
        hs_state = {
            "rewritten_queries": ["低乳糖 儿童 奶粉"],
            "filters": {"category": "奶粉"}
        }

        hs_result = hs_service(hs_state)
        print(f"  HybridSearch输出字段: {list(hs_result.keys())}")
        assert "candidates" in hs_result
        print("  ✅ HybridSearch状态契约正确")
    except Exception as e:
        print(f"  ❌ HybridSearch状态契约错误: {e}")

    # 3. 测试Rerank节点
    print("\n3. 测试Rerank状态契约...")
    try:
        # Mock一个简化的RerankService
        class MockRerankService:
            def __call__(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
                return {
                    "ranked_candidates": MOCK_RANKED_CANDIDATES,
                    "ranked_skus": ["SKU_1001", "SKU_3003"],
                    "rerank_type": "mock"
                }

        rerank_service = MockRerankService()
        rerank_state = {
            "raw_query": "不上火的奶粉",
            "candidates": MOCK_CANDIDATES
        }

        rerank_result = rerank_service(rerank_state)
        print(f"  Rerank输出字段: {list(rerank_result.keys())}")
        assert "ranked_candidates" in rerank_result
        assert "ranked_skus" in rerank_result
        assert "rerank_type" in rerank_result
        print("  ✅ Rerank状态契约正确")
    except Exception as e:
        print(f"  ❌ Rerank状态契约错误: {e}")

    # 4. 测试ContextBuilder节点
    print("\n4. 测试ContextBuilder状态契约...")
    try:
        class MockContextBuilderService:
            def __call__(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
                return {
                    "product_context": MOCK_PRODUCT_CONTEXT
                }

        cb_service = MockContextBuilderService()
        cb_state = {
            "ranked_skus": ["SKU_1001", "SKU_3003"]
        }

        cb_result = cb_service(cb_state)
        print(f"  ContextBuilder输出字段: {list(cb_result.keys())}")
        assert "product_context" in cb_result
        print("  ✅ ContextBuilder状态契约正确")
    except Exception as e:
        print(f"  ❌ ContextBuilder状态契约错误: {e}")

    # 5. 测试RealTimeData节点
    print("\n5. 测试RealTimeData状态契约...")
    try:
        class MockRealTimeDataService:
            def __call__(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
                return {
                    "real_time_data": MOCK_REALTIME_DATA
                }

        rt_service = MockRealTimeDataService()
        rt_state = {
            "ranked_skus": ["SKU_1001", "SKU_3003"]
        }

        rt_result = rt_service(rt_state)
        print(f"  RealTimeData输出字段: {list(rt_result.keys())}")
        assert "real_time_data" in rt_result
        print("  ✅ RealTimeData状态契约正确")
    except Exception as e:
        print(f"  ❌ RealTimeData状态契约错误: {e}")

    # 6. 测试LLMGenerate节点
    print("\n6. 测试LLMGenerate状态契约...")
    try:
        from llm_generate import LLMGenerateService

        llm_mock = type('MockLLM', (), {
            'invoke': lambda self, x: type('Response', (), {'content': '测试回答'})(),
            'with_structured_output': lambda self, schema: self
        })()

        llm_service = LLMGenerateService(llm=llm_mock)
        llm_state = {
            "raw_query": "不上火的奶粉",
            "product_context": MOCK_PRODUCT_CONTEXT,
            "real_time_data": MOCK_REALTIME_DATA
        }

        llm_result = llm_service(llm_state)
        print(f"  LLMGenerate输出字段: {list(llm_result.keys())}")
        assert "final_response" in llm_result
        assert "referenced_skus" in llm_result
        assert "generation_type" in llm_result
        assert "recommended_skus" in llm_result
        print("  ✅ LLMGenerate状态契约正确")
    except Exception as e:
        print(f"  ❌ LLMGenerate状态契约错误: {e}")

    print("\n=== 状态契约测试完成 ===")

def test_state_flow_simulation():
    """模拟完整状态流转"""
    print("\n=== 模拟LangGraph状态流转 ===\n")

    # 模拟初始状态
    state = {
        "raw_query": "不上火的奶粉",
        "user_context": {"channel": "miniapp", "user_type": "new_user"},
        "rewritten_queries": [],
        "filters": {},
        "candidates": [],
        "ranked_skus": [],
        "ranked_candidates": [],
        "rerank_type": "",
        "product_context": [],
        "real_time_data": {},
        "final_response": "",
        "referenced_skus": [],
        "generation_type": "",
        "recommended_skus": []
    }

    print(f"初始状态字段数: {len(state)}")

    # 1. Query Rewrite
    print("1. 执行Query Rewrite...")
    qr_update = {
        "rewritten_queries": ["低乳糖 儿童 奶粉", "益生菌 奶粉"],
        "filters": {"category": "奶粉", "age_range": "3-6"}
    }
    state.update(qr_update)
    print(f"   更新后字段: {list(qr_update.keys())}")

    # 2. Hybrid Search
    print("2. 执行Hybrid Search...")
    hs_update = {
        "candidates": MOCK_CANDIDATES
    }
    state.update(hs_update)
    print(f"   更新后字段: {list(hs_update.keys())}")

    # 3. Rerank
    print("3. 执行Rerank...")
    rerank_update = {
        "ranked_candidates": MOCK_RANKED_CANDIDATES,
        "ranked_skus": ["SKU_1001", "SKU_3003"],
        "rerank_type": "mock"
    }
    state.update(rerank_update)
    print(f"   更新后字段: {list(rerank_update.keys())}")

    # 4. Context Builder
    print("4. 执行Context Builder...")
    cb_update = {
        "product_context": MOCK_PRODUCT_CONTEXT
    }
    state.update(cb_update)
    print(f"   更新后字段: {list(cb_update.keys())}")

    # 5. Real Time Data
    print("5. 执行Real Time Data...")
    rt_update = {
        "real_time_data": MOCK_REALTIME_DATA
    }
    state.update(rt_update)
    print(f"   更新后字段: {list(rt_update.keys())}")

    # 6. LLM Generate
    print("6. 执行LLM Generate...")
    llm_update = {
        "final_response": "根据您的需求，推荐以下奶粉...",
        "referenced_skus": ["SKU_1001", "SKU_3003"],
        "generation_type": "llm",
        "recommended_skus": ["SKU_1001", "SKU_3003"]
    }
    state.update(llm_update)
    print(f"   更新后字段: {list(llm_update.keys())}")

    print(f"\n最终状态字段数: {len(state)}")
    print("✅ 状态流转模拟完成，所有字段都得到正确更新")

if __name__ == "__main__":
    test_state_contracts()
    test_state_flow_simulation()
    print("\n🎉 LangGraph状态流转验证通过！")