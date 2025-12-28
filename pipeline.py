"""
Module 7: LangGraph总编排服务
职责: 编排整个RAG流程
使用 LangGraph 1.0.4 API
"""
from typing import TypedDict, Annotated, Any
import json

try:
    from langgraph.graph import StateGraph, END, START
    from langchain_openai import ChatOpenAI

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    START = "START"
    END = "END"


    class StateGraph:
        def __init__(self, state_schema):
            pass


    class ChatOpenAI:
        pass

# 导入所有服务
from query_rewrite import QueryRewriteService
from hybrid_search import (
    VectorRetrievalService,
    KeywordRetrievalService,
    HybridSearchService
)
from rerank import create_reranker
from context_builder import ContextBuilderService
from realtime_data import RealTimeDataService
from llm_generate import LLMGenerateService
from models import PipelineState


class ProductRAGPipeline:
    """商品RAG完整流程编排 - 使用LangGraph 1.0.4"""

    def __init__(
            self,
            llm_model: str = "gpt-3.5-turbo",
            use_mock: bool = True
    ):
        """
        初始化RAG流程

        Args:
            llm_model: LLM模型名称
            use_mock: 是否使用Mock LLM (用于测试)
        """
        # 初始化LLM
        if use_mock:
            self.llm = self._create_mock_llm()
        else:
            self.llm = ChatOpenAI(model=llm_model, temperature=0)

        # 初始化所有服务
        print("初始化服务...")
        self.query_rewrite_service = QueryRewriteService(llm=self.llm)

        # 检索服务
        self.vector_service = VectorRetrievalService()
        self.keyword_service = KeywordRetrievalService()
        self.hybrid_search_service = HybridSearchService(
            self.vector_service,
            self.keyword_service
        )

        # 其他服务 - 使用工厂函数自动选择最佳重排策略
        self.rerank_service = create_reranker(top_n=5)
        self.context_builder_service = ContextBuilderService(max_tokens=2000)
        self.realtime_data_service = RealTimeDataService()
        self.llm_generate_service = LLMGenerateService(llm=self.llm)

        # 构建图
        self.graph = self._build_graph()
        print("流程图构建完成!")

    def _create_mock_llm(self):
        """创建Mock LLM用于测试"""
        import json

        class MockLLM:
            def __init__(self):
                self._structured_schemas = {}

            def with_structured_output(self, schema):
                """支持结构化输出的Mock实现"""
                class StructuredMockLLM:
                    def __init__(self, parent, schema):
                        self.parent = parent
                        self.schema = schema

                    def invoke(self, messages):
                        # 调用原始的invoke获得响应
                        response = self.parent.invoke(messages)
                        # 解析JSON并返回结构化对象
                        try:
                            import json
                            data = json.loads(response.content)
                            return self.schema(**data)
                        except:
                            # fallback到默认值
                            if hasattr(self.schema, '__annotations__'):
                                kwargs = {}
                                for field_name, field_type in self.schema.__annotations__.items():
                                    if field_type == str:
                                        kwargs[field_name] = response.content
                                    elif hasattr(field_type, '__origin__') and field_type.__origin__ == list:
                                        kwargs[field_name] = ["SKU_1001", "SKU_3003"]
                                    else:
                                        kwargs[field_name] = None
                                return self.schema(**kwargs)
                            return self.schema()

                return StructuredMockLLM(self, schema)

            def invoke(self, messages):
                # 根据消息内容判断是哪个服务调用
                message_text = str(messages)

                class Response:
                    content = ""

                response = Response()

                if "改写" in message_text or "rewrite" in message_text.lower():
                    # Query Rewrite
                    response.content = json.dumps({
                        "rewritten_queries": [
                            "低乳糖 儿童 奶粉",
                            "含益生菌 配方奶粉",
                            "肠胃友好型 奶粉"
                        ],
                        "filters": {
                            "category": "奶粉",
                            "age_range": "3-6",
                            "exclude_tags": ["成人"]
                        }
                    }, ensure_ascii=False)

                elif "排序" in message_text or "rank" in message_text.lower():
                    # Rerank
                    response.content = json.dumps({
                        "ranked_skus": ["SKU_1001", "SKU_3003", "SKU_5005"]
                    })

                elif "商品排序" in message_text or "排序" in message_text or "rank" in message_text.lower():
                    # LLM Generate 结构化输出
                    response.content = json.dumps({
                        "response": "根据您的需求,我为您推荐以下商品:\n\n1. **儿童低乳糖配方奶粉** (SKU_1001)",
                        "recommended_skus": ["SKU_1001", "SKU_3003", "SKU_5005"]
                    }, ensure_ascii=False)
                else:
                    # LLM Generate - 传统文本输出
                    response.content = """根据您的需求"不上火的奶粉",我为您推荐以下商品:

1. **儿童低乳糖配方奶粉** (SKU_1001)
   推荐理由: 采用低乳糖配方,特别适合肠胃敏感的儿童,添加益生菌促进消化,有效减少上火问题
   价格: ¥299 (满299减50)
   库存: 充足(120件)

2. **肠胃友好型配方奶粉** (SKU_3003)
   推荐理由: 特别添加益生元,呵护肠胃健康,减少上火,易消化配方适合3-6岁儿童
   价格: ¥279 (买2送1)
   库存: 充足(200件)

3. **无糖益生菌奶粉** (SKU_5005)
   推荐理由: 0蔗糖添加,10亿活性益生菌保护肠道,适合1-3岁幼儿
   价格: ¥329 (预售中)
   库存: 暂时缺货

建议优先选择第一款,低乳糖配方针对性更强,且当前有满减活动非常划算。如果预算有限,第二款也是不错的选择。"""

                return response

        return MockLLM()

    def _build_graph(self):
        """构建LangGraph工作流 - 使用LangGraph 1.0.4 API"""

        # 定义状态类型 (TypedDict for LangGraph 1.0.4) - 包含新的可观测性字段
        class GraphState(TypedDict):
            raw_query: str
            user_context: dict
            rewritten_queries: list
            filters: dict
            candidates: list
            ranked_skus: list
            ranked_candidates: list  # 新增: 完整的重排序结果
            rerank_type: str  # 新增: 重排序类型
            product_context: list
            real_time_data: dict
            final_response: str
            referenced_skus: list
            generation_type: str  # 新增: 生成类型
            recommended_skus: list  # 新增: LLM推荐的SKU

        # 创建图 (LangGraph 1.0.4)
        workflow = StateGraph(GraphState)

        # 添加节点 (使用lambda包装，确保每个节点是callable(state) -> partial_state)
        workflow.add_node("query_rewrite", lambda state: self.query_rewrite_service(state))
        workflow.add_node("hybrid_search", lambda state: self.hybrid_search_service(state))
        workflow.add_node("rerank", lambda state: self.rerank_service(state))
        workflow.add_node("context_builder", lambda state: self.context_builder_service(state))
        workflow.add_node("realtime_data", lambda state: self.realtime_data_service(state))
        workflow.add_node("llm_generate", lambda state: self.llm_generate_service(state))

        # 定义流程边 (使用LangGraph 1.0.4的新API)
        workflow.add_edge(START, "query_rewrite")
        workflow.add_edge("query_rewrite", "hybrid_search")
        workflow.add_edge("hybrid_search", "rerank")
        workflow.add_edge("rerank", "context_builder")
        workflow.add_edge("context_builder", "realtime_data")
        workflow.add_edge("realtime_data", "llm_generate")
        workflow.add_edge("llm_generate", END)

        # 编译图
        return workflow.compile()

    def run(self, query: str, user_context: dict = None) -> dict:
        """
        执行完整RAG流程

        Args:
            query: 用户查询
            user_context: 用户上下文

        Returns:
            完整结果字典
        """
        # 初始化状态 - 包含所有新字段
        initial_state = {
            "raw_query": query,
            "user_context": user_context or {},
            "rewritten_queries": [],
            "filters": {},
            "candidates": [],
            "ranked_skus": [],
            "ranked_candidates": [],  # 新增
            "rerank_type": "",  # 新增
            "product_context": [],
            "real_time_data": {},
            "final_response": "",
            "referenced_skus": [],
            "generation_type": "",  # 新增
            "recommended_skus": []  # 新增
        }

        # 执行图
        print(f"\n{'=' * 60}")
        print(f"开始处理查询: {query}")
        print(f"{'=' * 60}\n")

        result = self.graph.invoke(initial_state)

        return result

    # def visualize(self, output_path: str = "rag_pipeline.png"):
    #     """
    #     可视化流程图
    #
    #     Args:
    #         output_path: 输出图片路径
    #     """
    #     try:
    #         from IPython.display import Image, display
    #         display(Image(self.graph.get_graph().draw_mermaid_png()))
    #     except Exception as e:
    #         print(f"可视化失败: {e}")
    #         print("流程顺序: QueryRewrite → HybridSearch → Rerank → ContextBuilder → RealTimeData → LLMGenerate")


def main():
    """主函数 - 运行完整示例"""

    # 创建流程 (使用Mock LLM)
    pipeline = ProductRAGPipeline(use_mock=True)

    # 测试查询
    test_queries = [
        "不上火的奶粉",
        "适合3岁宝宝的有机奶粉",
        "含益生菌的配方奶粉"
    ]

    for query in test_queries:
        result = pipeline.run(
            query=query,
            user_context={
                "channel": "miniapp",
                "user_type": "new_user"
            }
        )

        print("\n" + "=" * 60)
        print("最终结果:")
        print("=" * 60)
        print(f"\n用户查询: {result['raw_query']}")
        print(f"\n改写查询: {result['rewritten_queries']}")
        print(f"\n过滤条件: {result['filters']}")
        print(f"\n候选商品数: {len(result['candidates'])}")
        print(f"\n排序后SKU: {result['ranked_skus']}")

        # 新增: 显示可观测性信息
        print(f"\n=== 可观测性信息 ===")
        print(f"重排序类型: {result.get('rerank_type', 'N/A')}")
        print(f"生成类型: {result.get('generation_type', 'N/A')}")
        print(f"LLM推荐SKU: {result.get('recommended_skus', 'N/A')}")
        print(f"实际引用SKU: {result['referenced_skus']}")

        if result.get('ranked_candidates'):
            print(f"\n重排序详细信息:")
            for i, candidate in enumerate(result['ranked_candidates'][:3], 1):
                if hasattr(candidate, 'sku_id') and hasattr(candidate, 'rerank_score'):
                    print(f"  {i}. {candidate.sku_id} (分数: {candidate.rerank_score:.3f})")
                elif isinstance(candidate, dict):
                    print(f"  {i}. {candidate.get('sku_id', 'N/A')} (分数: {candidate.get('rerank_score', 'N/A')})")

        print(f"\n最终回答:\n{result['final_response']}")
        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()